# Research on MPI and Checkpointing: A Comprehensive Overview

## Executive Summary

This report presents a comprehensive overview of research on the Message Passing Interface (MPI) and checkpointing techniques for fault tolerance in parallel computing environments. The research examines traditional checkpoint/restart approaches, modern implementations, and emerging fault tolerance paradigms. Key findings indicate that while checkpoint/restart has been a cornerstone of fault tolerance for MPI applications, the field is transitioning toward User Level Fault Mitigation (ULFM) approaches that offer greater flexibility and scalability for exascale computing systems.

## Introduction

As modern supercomputers scale to hundreds or thousands of individual nodes, the Message Passing Interface (MPI) remains the de facto programming model for parallel applications. However, the larger number of individual hardware components means that hardware faults are more likely to occur during long-running jobs. Users naturally want their programs to adapt to hardware faults and continue running, making fault tolerance a critical concern for MPI applications.

In the scope of MPI implementations, fault tolerance is defined as the ability to recover from one or more component failures in a well-defined manner with either a transparent or application-directed mechanism. Component failures may manifest as corrupted transmissions over faulty network interfaces or the failure of one or more processes due to processor or node failures.

## Fundamental Concepts

### Fault Tolerance as a Property

A common misconception about MPI is that the MPI Standard itself mandates that if any MPI process dies, all MPI processes in the job must die as well. This is not true. The basis for this misconception stems from the default error handler on MPI_COMM_WORLD being MPI_ERRORS_ARE_FATAL, which causes all processes to abort if any process fails. However, this can be changed to MPI_ERRORS_RETURN, allowing applications to handle errors themselves. Users can also define custom error handlers and attach them to specific communicators.

Fault tolerance is not a property of the MPI specification itself, nor of an MPI implementation alone, but rather a property of an MPI program coupled with an MPI implementation. The MPI Standard provides considerable flexibility in handling errors, which enables various approaches to achieving fault tolerance while allowing implementations to trade performance against fault tolerance to meet user needs.

### Requirements for Fault Tolerance

Most approaches to fault tolerance share a similar set of requirements. First, failures must be detectable by the system or application. Second, information (state) needed to continue the computation must be available, either through saved checkpoints or redundant computation. Third, the computation must be capable of being restarted from a consistent state.

## Checkpointing Approaches

### Overview of Checkpointing

Checkpointing is a common technique that periodically saves the state of a computation, allowing the computation to be restarted from that point in the event of failure. While often considered expensive, the cost of checkpointing need not be large when properly implemented. For small probability of failure (α) and relatively small costs of creating (k₀) and restoring (k₁) checkpoints, the added cost is quite modest. The optimal time between checkpoints can be calculated as t₀ = √(2k₀/α), and the expected total runtime becomes E_T = T(1 + √(αk₁) + √(2αk₀)).

The practicality of checkpointing is closely related to parallel I/O performance. MPI provides excellent facilities for performing I/O, including support for output in canonical form that can be used to restart computations on different numbers of processors.

### Coordinated Checkpointing

Coordinated checkpointing is the most widely used approach. It stores a snapshot of all processes in the distributed system such that the global application checkpoint is guaranteed to be consistent. This is achieved through orchestrated cooperation among all participating processes in determining their individual local checkpoints. The resulting set of local checkpoints constitutes a global checkpoint that represents a consistent global state of the application.

A consistent global state combines the states of all individual processes and all communication channels at any moment during correct, failure-free execution. Some algorithms, such as the Chandy-Lamport algorithm, attempt to empty the network from in-transit messages by preventing processes from sending messages and waiting to ensure all sent messages have reached their destinations. Each process stores a single checkpoint, significantly reducing storage overhead.

Recovery using coordinated checkpointing is simple. If failure occurs, all processes roll back to the most recent checkpoint, even when only one process fails. While coordination protocols consume more time computing the global checkpoint, most scientific programs are naturally iterative, allowing checkpoint computing to be performed while progressing from one iteration to the next, which guarantees a minimum checkpoint size.

### Uncoordinated Checkpointing

In uncoordinated checkpointing, all participating process checkpoints are independent from each other. These techniques generally rely on logging messages and possibly their temporal ordering for asynchronous checkpointing. Message-logging techniques and process checkpointing techniques together can completely describe the execution state of any process.

The main advantage is that checkpoints can be taken at the most convenient time for each process, reducing overhead. A process can perform checkpoints when its state is small. However, each process needs to maintain multiple checkpoints, increasing storage overhead. The main disadvantage is difficulty obtaining a global consistent state, making checkpoints potentially ineffective. Uncoordinated checkpointing is prone to the domino effect caused by rollback propagation from one process to another, which might continue until reaching the beginning of execution, leading to undesirable loss of computational work.

### Communication Induced Checkpointing (CIC)

Communication Induced Checkpointing combines aspects of uncoordinated and coordinated techniques. CIC piggybacks message reception orders on all application messages causally, guaranteeing an overall order of message delivery. It enforces chosen processes to be checkpointed when an inconsistent state is suspected, preserving recovery line progress.

Each process can select its most suitable time for checkpointing, such as when process state is small, reducing saving overhead. CIC protocols can avoid the domino effect. However, they are impractical due to considerable impact on network communication and application performance from managing piggybacked data. CIC does not scale well with increasing numbers of processes, leading to increasing numbers of forced checkpoints and increasing storage requirements.

## Implementation Levels

Checkpoint/restart systems can be integrated at three distinct levels, each with different tradeoffs between transparency, portability, and efficiency.

### System-Level Implementation

System-level implementations include checkpoint/restart procedures in the operating system kernel, usually as a dynamically loaded kernel module. This approach is always transparent to users since no program changes are required. However, kernel-level implementations are not portable to other platforms and require kernel source code availability for modification, which is not always possible.

### User-Level Implementation

User-level implementations provide checkpoint/restart functionality through a library, and applications must be linked to this library. Because of necessary explicit linking, this approach is usually not transparent to users, depending on implementation details. One drawback is that the library must be permitted to make system calls to access system data, which is not always available.

### Application-Level Implementation

Application-level implementations involve writing all checkpoint/restart activities into the application or directly injecting them into application code using an automated preprocessor. These implementations are not transparent but provide more control over checkpoint/restart processes, requiring very deep understanding of application details.

## Case Study: Checkpointing over InfiniBand

### Challenges

Checkpointing MPI applications over InfiniBand presents unique challenges compared to traditional TCP/IP-based networks. InfiniBand provides high-performance communication via OS-bypass capabilities in its user-level protocol. Unlike TCP/IP networks where the operating system kernel handles all network activities, InfiniBand skips the OS in actual communication, creating an information gap between the OS kernel and the user-land application process.

InfiniBand network adapters store network connection context in adapter memory, which is designed to be volatile. This information cannot be easily reused by restarted processes. Therefore, network connection context must be released before checkpointing and rebuilt afterward. Some context information, such as Queue Pairs (QPs), is also cached in user memory and must be reconstructed according to the new network connection context.

Additionally, many applications use RDMA operations provided by InfiniBand. InfiniBand requires authentication for accessing remote memory through registered memory regions and remote keys. These keys become invalid when network connection context is rebuilt, potentially introducing inconsistency.

### Framework Architecture

The checkpoint/restart framework for InfiniBand-based MPI applications, implemented in MVAPICH2, consists of five key components. The Global C/R Coordinator manages checkpoint/restart for the entire MPI job and can be configured to initiate checkpoints periodically or handle requests from users or administrators. The Control Message Manager provides an interface between the global coordinator and local controllers using out-of-band messaging.

Local C/R Controllers manage C/R operations for each MPI process, taking requests from the global coordinator, cooperating with communication channel managers and peer controllers to converge the job to a consistently checkpointable state, and invoking the C/R library for local checkpoints. The C/R Library handles checkpointing and restarting individual processes, with implementations using packages like Berkeley Lab's Checkpoint/Restart (BLCR). Finally, Communication Channel Managers control in-band message passing with extended functionalities for suspending and reactivating communication channels transparently.

### Performance Results

Experimental results using NAS benchmarks, HPL benchmark, and GROMACS demonstrated that checkpoint overhead is low and performance impact is insignificant. For GROMACS, the time for checkpointing was less than 0.3% of execution time, and performance decreased by only 4% with checkpoints taken every minute. This demonstrates that application-transparent checkpointing can be achieved with minimal performance impact when properly designed.

## Modern Checkpoint/Restart Tools

### BLCR (Berkeley Lab Checkpoint/Restart)

BLCR is a transparent system-level kernel-based infrastructure for checkpoint/restart. It is open source and deployed on several distributed systems. BLCR stores checkpoint data including stack, heap, registers, and signals. It can checkpoint/restart a single node as a standalone system or parallel applications running on multiple nodes by adding it as an additional configuration to a parallel communication library or scheduling system.

BLCR supports serial and multithreaded applications on x86 and x86_64 systems with Linux kernels from 2.6.x through 3.7.1. It does not checkpoint or restart open files or sockets like TCP/UDP. OpenMPI has integrated BLCR to support distributed coordinated checkpoint/restart fault tolerance within its modular component architecture. BLCR is particularly notable for its widespread usage and simple interface that allows easy integration with libraries and applications.

### DMTCP (Distributed Multithreaded Checkpointing)

DMTCP is a freely available user-level coordinated checkpoint/restart library implementation for distributed computations. It supports sequential and multi-threaded computations across single or multiple hosts. DMTCP is transparent, requiring no recompilation or relinking. It works entirely in user space with no added kernel modules and no need for root privileges.

A dynamically injected library and checkpointing manager thread are spawned in each application process. DMTCP stores user space memory, processor state, kernel state, and network data. It can checkpoint/restart a wide range of applications including scripting languages (MATLAB, Python, PHP, Ruby) and distributed platforms (MPICH2, OpenMPI). Checkpoint/restart can be scheduled periodically or manually initiated.

DMTCP is a general lightweight solution for socket-based distributed applications that can checkpoint high-performance networks like InfiniBand. It has a two-layer architecture: the first layer handles checkpoint details of processes across networked nodes, copying inter-process data to user space; the second layer is a single-process checkpointer (MTCP). This architecture allows DMTCP to work in non-Linux environments by replacing the single-process checkpointer.

### Performance Comparison

Recent performance evaluations on Amazon EC2 cloud infrastructure compared DMTCP and BLCR across multiple dimensions. DMTCP performed better than BLCR for both checkpoint and restart speed. DMTCP showed better performance as data size increased, demonstrating superior data scalability. DMTCP also demonstrated better scalability as the number of processes increased. These findings help legacy MPI distributed application developers choose the suitable checkpoint/restart tool when migrating their work to cloud environments.

## Current State in Open MPI

Open MPI has been a vehicle for research in fault tolerance and over the years has provided support for a wide range of resilience techniques. However, the landscape has changed significantly in recent years.

### Deprecated Features

Support for coordinated and uncoordinated process checkpoint and restart, similar to those implemented in LAM/MPI and MPICH-V respectively, has been deprecated. Data reliability and network fault tolerance features, similar to those implemented in LA-MPI, have also been deprecated. These deprecations occurred due to lack of adoption and lack of maintenance. While traces are still available in the main repository for archeological purposes, checkpoint/restart is no longer actively maintained in current Open MPI versions.

### Current Focus

The only active work in resilience in Open MPI targets User Level Fault Mitigation (ULFM), a technique discussed in the context of the MPI standardization body. Message logging techniques, similar to those implemented in MPICH-V, remain available only for research and non-production usage.

## User Level Fault Mitigation (ULFM)

### Overview

User Level Fault Mitigation (ULFM) has emerged as the front-running solution for process fault tolerance in MPI. It is a proposal developed by the MPI Forum's Fault Tolerance Working Group to support the continued operation of MPI applications in the presence of failures. ULFM is becoming the preferred approach over traditional checkpoint/restart mechanisms and is being considered for inclusion in the MPI standard.

### Key Features

ULFM is a set of new interfaces for MPI that enables message-passing applications to restore MPI functionality affected by failures. It provides fault-tolerant semantics in MPI through new MPI operations that add fault tolerance functionalities. These include mechanisms for detecting process failures, notifying applications of failures, reconstructing communicators after failures, and enabling continued MPI operations after failures.

### Research and Adoption

Substantial research has demonstrated the viability and performance of ULFM. Studies have shown that while faulty communicator reconstruction time can be large, especially for multiple process failures, ULFM provides a flexible framework for application-level fault tolerance. Researchers have chosen ULFM for developing failure-mitigating versions of scientific applications, and it has been successfully applied to domains including bioinformatics and partial differential equation solvers.

### Comparison with Checkpoint/Restart

Traditional checkpoint/restart approaches periodically save complete application state and require restarting from the last checkpoint on failure. While they can be transparent to applications, they incur higher overhead for large-scale systems and have been deprecated in modern Open MPI.

In contrast, ULFM allows applications to detect and respond to failures, enabling continued operation without full restart. While it requires application-level fault tolerance logic, it is more flexible and scalable for exascale systems. ULFM represents the current active development direction in Open MPI and is being considered for standardization.

## Recent Developments

Recent research continues to advance checkpointing techniques. Work published in 2024 addresses the long-standing problem of developing a practical and general algorithm for transparent checkpointing of MPI that is both efficient and compatible with modern systems. Hybrid checkpointing approaches that alternate between full and incremental checkpoints have been explored to reduce checkpoint overhead by capturing only data changed since the last checkpoint.

Multi-level checkpoint/restart designs have demonstrated significant improvements, with some approaches reducing end-to-end recovery time by up to 360% when recovering 1.3 TB of checkpointed data. These advances show that while the focus has shifted toward ULFM, checkpoint/restart techniques continue to evolve and remain relevant for specific use cases.

## Conclusions

Research on MPI and checkpointing reveals a rich landscape of techniques, implementations, and ongoing developments. Traditional checkpoint/restart approaches have provided valuable fault tolerance capabilities for MPI applications, with coordinated checkpointing emerging as the most widely adopted technique. Modern implementations like DMTCP and BLCR have demonstrated that efficient, transparent checkpointing is achievable with minimal performance overhead.

However, the field is transitioning toward User Level Fault Mitigation (ULFM) as the primary fault tolerance mechanism for MPI applications. This shift reflects the challenges of scaling traditional checkpoint/restart to exascale systems and the need for more flexible, application-aware fault tolerance approaches. ULFM provides a framework that allows applications to detect and respond to failures while continuing operation, offering better scalability and flexibility than periodic checkpoint/restart.

For practitioners, the choice between checkpoint/restart and ULFM depends on specific application requirements, system characteristics, and the desired level of transparency versus control. Checkpoint/restart remains valuable for applications where transparent fault tolerance is essential and checkpoint overhead is acceptable. ULFM is more suitable for applications that can incorporate fault tolerance logic and require fine-grained control over failure handling.

As MPI applications continue to scale to larger systems, fault tolerance will remain a critical research area. The ongoing development of ULFM and its potential standardization represent important steps toward providing robust, scalable fault tolerance for the next generation of high-performance computing applications.

## Key Research Papers

### Foundational Work

1. **Fault Tolerance in MPI Programs** by William Gropp and Ewing Lusk (Cited by 264)  
   Argonne National Laboratory  
   https://www.mcs.anl.gov/~lusk/papers/fault-tolerance.pdf

2. **Application-Transparent Checkpoint/Restart for MPI Programs over InfiniBand** by Qi Gao, Weikuan Yu, Wei Huang, Dhabaleswar K. Panda (Cited by 111)  
   The Ohio State University, 2006  
   https://mvapich.cse.ohio-state.edu/static/media/publications/abstract/gaoq-icpp06.pdf

3. **Checkpointing message-passing interface (MPI) parallel programs** by WJ Li and JJ Tsay (Cited by 31)  
   1997  
   https://ieeexplore.ieee.org/abstract/document/640140/

### Modern Implementations

4. **Performance Evaluation of Checkpoint/Restart Techniques for MPI Applications on Amazon Cloud** by Basma Abdel Azeem and Manal Helal (2023)  
   https://arxiv.org/pdf/2311.17545

5. **Hybrid Checkpointing for MPI Jobs in HPC Environments** by C. Wang (Cited by 72)  
   https://arcb.csc.ncsu.edu/~mueller/ftp/pub/mueller/papers/icpads10-2.pdf

### ULFM and Modern Approaches

6. **Lessons Learned Implementing User-Level Failure Mitigation** by W. Bland (Cited by 25)  
   https://pavanbalaji.github.io/pubs/2015/ccgrid/ccgrid15.ulfm.pdf

7. **Evaluating User-Level Fault Tolerance for MPI Applications** by I. Laguna (Cited by 58)  
   2014  
   https://dl.acm.org/doi/10.1145/2642769.2642775

8. **Fault tolerance of MPI applications in exascale systems** by N. Losada (Cited by 65)  
   2020  
   https://www.sciencedirect.com/science/article/pii/S0167739X1930860X

9. **Using Fault-Tolerant Open MPI in a PDE Solver** by MM Ali (Cited by 44)  
   2014  
   https://ieeexplore.ieee.org/document/6969514/

10. **Enabling Practical Transparent Checkpointing for MPI** (2024)  
    https://arxiv.org/html/2408.02218v1

## Additional Resources

- **Open MPI Fault Tolerance Documentation**  
  https://docs.open-mpi.org/en/main/tuning-apps/fault-tolerance/

- **Open MPI ULFM Documentation**  
  https://docs.open-mpi.org/en/main/features/ulfm.html

- **Fault Tolerance Research Hub**  
  https://fault-tolerance.org/

- **ULFM Project at ICL**  
  https://icl.utk.edu/research/ulfm

- **ULFM Testing Repository**  
  https://github.com/ICLDisco/ulfm-testing

- **MPI Forum's Fault Tolerance Working Group**  
  Information available through the MPI Forum website
