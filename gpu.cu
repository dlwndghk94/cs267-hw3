#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256
#define cutoff  0.01


extern double size;
//
//  benchmarking program
//

__global__ void init_bins_gpu(bin_t *bins, int num_bin){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_bin) return;

    bins[tid].head = -1;
}

__global__ void assign_particles_to_bins_gpu(particle_t *particles, bin_t *bins, int n, int bin_dim) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  // find which bin the particle belongs to
  int i = particles[tid].x / bin_dim;
  int j = particles[tid].y / bin_dim;
  int idx = j*bin_dim + i; 

  // assign particles
  particles[tid].next = atomicExch(&bins[idx].head, tid);
}


__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__device__ void compute_forces_particle_and_bin(int particle_idx, int bin_idx, particle_t *particles, bin_t *bins)
{
  int head_idx = bins[bin_idx].head;
  while (head_idx != -1) {
    apply_force_gpu(particles[particle_idx], particles[head_idx]);
    head_idx = particles[head_idx].next;
  }
}

__global__ void compute_forces_gpu(particle_t *particles, bin_t *bins, int n, int bin_dim)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;

  // find which bin the particle belongs to
  int i = particles[tid].x / bin_dim;
  int j = particles[tid].y / bin_dim;

  // find up, down, left, right, curr bins
  int lower_x_lim = (i > 0) ? (i - 1) : 0;
  int lower_y_lim = (j > 0) ? (j - 1) : 0;
  int upper_x_lim = (i < bin_dim - 1) ? (i + 1) : (bin_dim - 1);
  int upper_y_lim = (j < bin_dim - 1) ? (j + 1) : (bin_dim - 1);

  // apply force between particle and all particles in surrounding bins
  for (int yy = lower_y_lim; yy <= upper_y_lim; yy++) {
    for (int xx = lower_x_lim; xx <= upper_x_lim; xx++) {
      compute_forces_particle_and_bin(tid, yy*bin_dim+xx, particles, bins);
    }
  }
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

    p->next = NULL;

}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    // setup memory for bins for GPU
    int bin_dim = size/cutoff; // the dimension (size) of a bin
    int num_bins = bin_dim * bin_dim;
    bin_t *d_bins;
    cudaMalloc((void **) &d_bins, num_bins * sizeof(bin_t));
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    int bin_blks = (num_bins + NUM_THREADS - 1) / NUM_THREADS;  // GPU block for bins
    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;             // GPU block for particles
    for( int step = 0; step < NSTEPS; step++ )
    { 
      //  
      // Initialize/reinitialize bins
      //
      printf("step %i, initialize bins necessary\n", step);
      init_bins_gpu<<< bin_blks, NUM_THREADS >>> (d_bins, num_bins);

      //
      // Assign particles to bins
      //  
      printf("step %i, assign particles to bins\n", step);
      assign_particles_to_bins_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, n, bin_dim);    

      //
      //  compute forces
      //
      printf("step %i, compute forces\n", step);
	    compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, n, bin_dim);
        
      //
      //  move particles
      //
      printf("step %i, move particles\n", step);
	    move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
      //
      //  save if necessary
      //
      if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
        cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
        save( fsave, n, particles);
	    }

    }

    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
