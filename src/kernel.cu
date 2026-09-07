#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 12.0f
#define rule2Distance 4.0f
#define rule3Distance 9.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_posSorted;
glm::vec3 *dev_vel1Sorted;
glm::vec3 *dev_vel2Sorted;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  // 2.1
  cudaMalloc((void**)&dev_particleArrayIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMalloc((void**)&dev_particleGridIndices, numObjects * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // 2.3
  cudaMalloc((void**)&dev_posSorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posSorted failed!");
  cudaMalloc((void**)&dev_vel1Sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1Sorted failed!");
  cudaMalloc((void**)&dev_vel2Sorted, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2Sorted failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

__device__ glm::vec3 clampSpeed(glm::vec3 vel) {
  float speed = glm::length(vel);
  if (speed > maxSpeed) {
     vel = (vel / speed) * maxSpeed;
  }
  return vel;
}


__device__ glm::vec3 rule1Naive(int N, int iSelf, const glm::vec3 *pos) {
  glm::vec3 selfPos = pos[iSelf];

  // Calc avg neighbor position
  int total = 0;
  glm::vec3 avgPos = glm::vec3(0.0f);
  for(int t=0; t<N; t++) {
    glm::vec3 otherPos = pos[t];
    if (t != iSelf && glm::distance(selfPos, otherPos) < rule1Distance) {
      avgPos += pos[t];
      total++;
    }
  }

  // Edge case- no neighbors
  if (total == 0) {
    return glm::vec3(0.0f);
  }

  // Avg and calc vel
  avgPos /= total;
  return (avgPos - selfPos) * rule1Scale;
}

__device__ glm::vec3 rule2Naive(int N, int iSelf, const glm::vec3 *pos) {
  glm::vec3 selfPos = pos[iSelf];

  glm::vec3 total = glm::vec3(0.0f);
  for (int t=0; t<N; t++) {
    glm::vec3 otherPos = pos[t];
    if (t != iSelf && glm::distance(selfPos, otherPos) < rule2Distance) {
      total -= pos[t] - selfPos;
    }
  }

  return total * rule2Scale;
}

__device__ glm::vec3 rule3Naive(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 selfPos = pos[iSelf];

  // Calc avg neighbor position
  int total = 0;
  glm::vec3 avgVel = glm::vec3(0.0f);
  for(int t=0; t<N; t++) {
    glm::vec3 otherPos = pos[t];
    if (t != iSelf && glm::distance(selfPos, otherPos) < rule3Distance) {
      avgVel += vel[t];
      total++;
    }
  }

  // Edge case- no neighbors
  if (total == 0) {
    return glm::vec3(0.0f);
  }

  // Avg and calc vel
  avgVel /= total;
  return avgVel * rule3Scale;
}

struct BoidRuleAccum {
    glm::vec3 rule1center = glm::vec3(0.f);
    glm::vec3 rule2center = glm::vec3(0.f);
    glm::vec3 rule3avgVel = glm::vec3(0.f);
    int rule1total = 0;
    int rule3total = 0;
};

__device__ void rule1ScatteredGrid(const glm::vec3 &selfPos, const glm::vec3 &otherPos, BoidRuleAccum &boidRuleAccum) {
  if (glm::distance(selfPos, otherPos) < rule1Distance) {
    boidRuleAccum.rule1center += otherPos;
    boidRuleAccum.rule1total++;
  }
}

__device__ void rule2ScatteredGrid(const glm::vec3 &selfPos, const glm::vec3 &otherPos, BoidRuleAccum &boidRuleAccum) {
  if (glm::distance(selfPos, otherPos) < rule2Distance) {
    boidRuleAccum.rule2center -= otherPos - selfPos;
  }
}

__device__ void rule3ScatteredGrid(const glm::vec3 &selfPos, const glm::vec3 &otherPos, const glm::vec3 &otherVel, BoidRuleAccum &boidRuleAccum) {
  if (glm::distance(selfPos, otherPos) < rule3Distance) {
    boidRuleAccum.rule3avgVel += otherVel;
    boidRuleAccum.rule3total++;
  }
}

__device__ void finalizeVelocity(const glm::vec3 &selfPos, glm::vec3 &selfVel, BoidRuleAccum &boidRuleAccum) {
  // Finalize rule 1
  if (boidRuleAccum.rule1total > 0) {
    boidRuleAccum.rule1center /= boidRuleAccum.rule1total;
    selfVel += (boidRuleAccum.rule1center - selfPos) * rule1Scale;
  }

  // Finalize rule 2
  selfVel += boidRuleAccum.rule2center * rule2Scale;

  // Finalize rule 3
  if (boidRuleAccum.rule3total > 0) {
    boidRuleAccum.rule3avgVel /= boidRuleAccum.rule3total;
    selfVel += boidRuleAccum.rule3avgVel * rule3Scale;
  }
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 rule1vel = rule1Naive(N, iSelf, pos);

  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 rule2vel = rule2Naive(N, iSelf, pos);

  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 rule3vel = rule3Naive(N, iSelf, pos, vel);

  return rule1vel + rule2vel + rule3vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
  if (iSelf >= N) {
    return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 newVel = vel1[iSelf] + computeVelocityChange(N, iSelf, pos, vel1);

  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[iSelf] = clampSpeed(newVel);
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    // Thread index
    int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iSelf >= N) {
      return;
    }

    // Relative position (removes worldspace offset)
    glm::vec3 relPos = pos[iSelf] - gridMin;

    // Get cell index in each dimension
    int cellX = int(relPos.x * inverseCellWidth);
    int cellY = int(relPos.y * inverseCellWidth);
    int cellZ = int(relPos.z * inverseCellWidth);

    // Convert to flat index
    int gridIndex = gridIndex3Dto1D(cellX, cellY, cellZ, gridResolution);

    // Record to gridIndices
    gridIndices[iSelf] = gridIndex;
    indices[iSelf] = iSelf;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    // Thread index
    int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iSelf >= N) {
      return;
    }

    // Check if start
    if (iSelf == 0 ) { // Boid 0 is always start
      gridCellStartIndices[particleGridIndices[iSelf]] = iSelf;
    }
    else if (particleGridIndices[iSelf - 1] != particleGridIndices[iSelf]) {
      gridCellStartIndices[particleGridIndices[iSelf]] = iSelf;
    }

    // Check if end
    if (iSelf == N - 1) { // Boid N-1 is always end
      gridCellEndIndices[particleGridIndices[iSelf]] = iSelf;
    }
    else if (particleGridIndices[iSelf] != particleGridIndices[iSelf + 1]) {
      gridCellEndIndices[particleGridIndices[iSelf]] = iSelf;
    }
}

__device__ void getAdjacentCells8(glm::vec3 relPos, float inverseCellWidth, float cellWidth, int gridResolution, int *adjCells) {
  int cellX = int(relPos.x * inverseCellWidth);
  int cellY = int(relPos.y * inverseCellWidth);
  int cellZ = int(relPos.z * inverseCellWidth);

  // Check if we round up or down in each dimension
  int xDir = (relPos.x - cellX * cellWidth < cellWidth / 2.0f) ? -1 : 1;
  int yDir = (relPos.y - cellY * cellWidth < cellWidth / 2.0f) ? -1 : 1;
  int zDir = (relPos.z - cellZ * cellWidth < cellWidth / 2.0f) ? -1 : 1;

  // Check current + adjacent cell in each dimension
  int xCells[2] = {cellX, cellX + xDir};
  int yCells[2] = {cellY, cellY + yDir};
  int zCells[2] = {cellZ, cellZ + zDir};

   // Check adjacent 8 cells
  for (int i=0; i<=1; i++) {
    for (int j=0; j<=1; j++) {
      for (int k=0; k<=1; k++) {
        int x = xCells[i];
        int y = yCells[j];
        int z = zCells[k];

        // Skip if out of bounds
        if (x < 0 || x >= gridResolution || 
            y < 0 || y >= gridResolution || 
            z < 0 || z >= gridResolution) 
            continue;
        
        // Record adjacent index
        int adjIndex = gridIndex3Dto1D(x, y, z, gridResolution);
        adjCells[gridIndex3Dto1D(i, j, k, 2)] = adjIndex;
      }
    }
  }
}

__device__ void getAdjacentCells27(glm::vec3 relPos, float inverseCellWidth, int gridResolution, int *adjCells) {
  int cellX = int(relPos.x * inverseCellWidth);
  int cellY = int(relPos.y * inverseCellWidth);
  int cellZ = int(relPos.z * inverseCellWidth);
  
  // Check adjacent 27 cells
  for (int i=-1; i<=1; i++) {
    for (int j=-1; j<=1; j++) {
      for (int k=-1; k<=1; k++) {
        int x = cellX + i;
        int y = cellY + j;
        int z = cellZ + k;

        // Skip if out of bounds
        if (x < 0 || x >= gridResolution || 
            y < 0 || y >= gridResolution || 
            z < 0 || z >= gridResolution) 
            continue;

        // Record adjacent index
        int adjIndex = gridIndex3Dto1D(x, y, z, gridResolution);
        adjCells[gridIndex3Dto1D(i + 1, j + 1, k + 1, 3)] = adjIndex;
      }
    }
  }
}

// Fills adjCells (either 8 or 27 cells)
__device__ void getAdjacentCells(glm::vec3 relPos, float inverseCellWidth, float cellWidth, int gridResolution, bool use27, int *adjCells) {
  if (use27) {
    getAdjacentCells27(relPos, inverseCellWidth, gridResolution, adjCells);
    return;
  }
  getAdjacentCells8(relPos, inverseCellWidth, cellWidth, gridResolution, adjCells);
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices, int *particleGridIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  // Get thread index
  int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
  if (iSelf >= N) return;
  
  // Get boid index (in pos/vel arrays)
  int boidIndex = particleArrayIndices[iSelf];
  glm::vec3 selfPos = pos[boidIndex];
  glm::vec3 selfVel = vel1[boidIndex];

  // Get adjacent cells
  const int MAX_ADJ_CELLS = 27;
  int adjacentCellIndices[MAX_ADJ_CELLS];
  for (int i=0; i<MAX_ADJ_CELLS; i++) {
    adjacentCellIndices[i] = -1;
  }
  bool use27 = false; // Use either 27 or 8 adj cells
  getAdjacentCells(selfPos - gridMin, inverseCellWidth, cellWidth, gridResolution, use27, adjacentCellIndices);

  // Init rules values
  BoidRuleAccum boidRuleAccum;

  // Loop through adjacent cells
  for (int i=0; i<MAX_ADJ_CELLS; i++) {
    int adjIndex = adjacentCellIndices[i];
    if (adjIndex == -1) continue;

    // Iterate through boids in each cell
    int cellStart = gridCellStartIndices[adjIndex];
    int cellEnd = gridCellEndIndices[adjIndex];
    for(int j=cellStart; j<=cellEnd; j++) {

      // Convert from sorted index -> original index
      int otherBoidIndex = particleArrayIndices[j];
      if (otherBoidIndex == boidIndex) continue;

      glm::vec3 otherPos = pos[otherBoidIndex];
      glm::vec3 otherVel = vel1[otherBoidIndex];

      rule1ScatteredGrid(selfPos, otherPos, boidRuleAccum);
      rule2ScatteredGrid(selfPos, otherPos, boidRuleAccum);
      rule3ScatteredGrid(selfPos, otherPos, otherVel, boidRuleAccum);
    }
  }

  // Compute new velocity
  finalizeVelocity(selfPos, selfVel, boidRuleAccum);

  // Record the new velocity into vel2
  vel2[boidIndex] = clampSpeed(selfVel);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  // Get thread index
  int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
  if (iSelf >= N) return;
  
  glm::vec3 selfPos = pos[iSelf];
  glm::vec3 selfVel = vel1[iSelf];

  // Get adjacent cells
  const int MAX_ADJ_CELLS = 27;
  int adjacentCellIndices[MAX_ADJ_CELLS];
  for (int i=0; i<MAX_ADJ_CELLS; i++) {
    adjacentCellIndices[i] = -1;
  }
  bool use27 = false; // Use either 27 or 8 adj cells
  getAdjacentCells(selfPos - gridMin, inverseCellWidth, cellWidth, gridResolution, use27, adjacentCellIndices);

  // Init rules values
  BoidRuleAccum boidRuleAccum;

  // Loop through adjacent cells
  for (int i=0; i<MAX_ADJ_CELLS; i++) {
    int adjIndex = adjacentCellIndices[i];
    if (adjIndex == -1) continue;

    // Iterate through boids in each cell
    int cellStart = gridCellStartIndices[adjIndex];
    int cellEnd = gridCellEndIndices[adjIndex];
    for(int j=cellStart; j<=cellEnd; j++) {

      // Convert from sorted index -> original index
      if (j == iSelf) continue;

      glm::vec3 otherPos = pos[j];
      glm::vec3 otherVel = vel1[j];

      rule1ScatteredGrid(selfPos, otherPos, boidRuleAccum);
      rule2ScatteredGrid(selfPos, otherPos, boidRuleAccum);
      rule3ScatteredGrid(selfPos, otherPos, otherVel, boidRuleAccum);
    }
  }

  // Compute new velocity
  finalizeVelocity(selfPos, selfVel, boidRuleAccum);

  // Record the new velocity into vel2
  vel2[iSelf] = clampSpeed(selfVel);
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // TODO-1.2 ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  // Reset buffers from last step
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<fullBlocksPerGridCells, blockSize>>>(
    gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullBlocksPerGridCells, blockSize>>>(
    gridCellCount, dev_gridCellEndIndices, -1);
  
  // Label each boid's cell index
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // Sort keys using Thrust 
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // (grid indices are now sorted [0->cellCnt) ascending)
  // dev_particleArrayIndices is now REORDERED!

  // Find start/end of grid indices array
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // Update velocity/pos
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
    dev_gridCellStartIndices, dev_gridCellEndIndices, 
    dev_particleArrayIndices, dev_particleGridIndices,
    dev_pos, dev_vel1, dev_vel2);
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // Ping pong velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

__global__ void kernSortParticleData(
  int N, int *sortedParticleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, 
  glm::vec3 *sortedPos, glm::vec3 *sortedVel1) {

  // Get thread index
  int iSelf = blockDim.x * blockIdx.x + threadIdx.x;
  if (iSelf >= N) return;

  // Set sorted pos/vel1 so that they line up w/ sorted indices
  int ogIndex = sortedParticleArrayIndices[iSelf];
  sortedPos[iSelf] = pos[ogIndex];
  sortedVel1[iSelf] = vel1[ogIndex];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  // Reset buffers from last step
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<fullBlocksPerGridCells, blockSize>>>(
    gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullBlocksPerGridCells, blockSize>>>(
    gridCellCount, dev_gridCellEndIndices, -1);
  
  // Label each boid's cell index
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // Sort keys using Thrust 
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
  // (grid indices are now sorted [0->cellCnt) ascending)
  // dev_particleArrayIndices is now REORDERED!

  // Find start/end of grid indices array
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // DIFFERENCE! Reshuffle velocity/ position arrays to match grid indices
  kernSortParticleData<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_posSorted, dev_vel1Sorted
  );

  // Update velocity/pos
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
    dev_gridCellStartIndices, dev_gridCellEndIndices, 
    dev_posSorted, dev_vel1Sorted, dev_vel2Sorted);

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_posSorted, dev_vel2Sorted);

  // Ping pong velocity buffers
  std::swap(dev_pos, dev_posSorted);
  std::swap(dev_vel1, dev_vel2Sorted);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices); 
  
  cudaFree(dev_posSorted);
  cudaFree(dev_vel1Sorted);
  cudaFree(dev_vel2Sorted);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
