//! Basic MPI test to verify MPI initialization

#[cfg(feature = "mpi")]
#[test]
fn test_mpi_basic() {
    use mpi::traits::*;

    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Process {}/{}: Hello from MPI!", rank, size);

    world.barrier();

    if rank == 0 {
        println!("All {} processes completed", size);
    }
}
