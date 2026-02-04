///! Minimal MPI hello world example
///! Run with: mpirun -n 4 cargo run --example mpi_hello --features mpi

fn main() {
    use mpi::traits::*;

    // Initialize MPI
    let universe = mpi::initialize().expect("Failed to initialize MPI");
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    println!("Hello from rank {}/{}", rank, size);

    world.barrier();

    if rank == 0 {
        println!("\nAll {} processes completed successfully", size);
    }
}
