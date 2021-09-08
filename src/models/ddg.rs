// “D. J. Kuck, R. H. Kuhn, D. A. Padua, B. Leasure, and M. Wolfe (1981). DEPENDENCE GRAPHS AND COMPILER OPTIMIZATIONS.”
//
// The current implementation of DDG differs slightly from the dependence graph described in [1] in the following ways:
//
// The graph nodes in the paper represent three main program components, namely assignment statements, for loop headers and while loop headers. In this implementation, DDG nodes naturally represent LLVM IR instructions. An assignment statement in this implementation typically involves a node representing the store instruction along with a number of individual nodes computing the right-hand-side of the assignment that connect to the store node via a def-use edge. The loop header instructions are not represented as special nodes in this implementation because they have limited uses and can be easily identified, for example, through LoopAnalysis.
// The paper describes five types of dependency edges between nodes namely loop dependency, flow-, anti-, output-, and input- dependencies. In this implementation memory edges represent the flow-, anti-, output-, and input- dependencies. However, loop dependencies are not made explicit, because they mainly represent association between a loop structure and the program elements inside the loop and this association is fairly obvious in LLVM IR itself.
// The paper describes two types of pi-blocks; recurrences whose bodies are SCCs and IN nodes whose bodies are not part of any SCC. In this implementation, pi-blocks are only created for recurrences. IN nodes remain as simple DDG nodes in the graph.

// https://llvm.org/docs/DependenceGraphs/index.html#id6

#[derive(Clone)]
pub struct DDG;
