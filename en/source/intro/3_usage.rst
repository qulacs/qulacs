Usage
-------------

| See the following document for more detail.

| :doc:`4.1_python_tutorial`.
| :doc:`4.2_cpp_tutorial`.
| `Examples`_.
| :doc:`../pyRef/modules`.
| :doc:`../api/cpp_library_root`.


C++ Libraries
~~~~~~~~~~~~~

Add ``./<qulacs_path>/include/`` to include path, and
``./<qulacs_path>/lib/`` to library path. If you use dynamic link
library, add ``./<qulacs_path>/bin/`` to library path instead.

Example of C++ code:

.. code:: cpp

   #include <iostream>
   #include <cppsim/state.hpp>
   #include <cppsim/circuit.hpp>
   #include <cppsim/observable.hpp>

   int main(){
       QuantumState state(3);
       state.set_Haar_random_state();

       QuantumCircuit circuit(3);
       circuit.add_X_gate(0);
       auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
       circuit.add_gate(merged_gate);
       circuit.add_RX_gate(1,0.5);
       circuit.update_quantum_state(&state);

       Observable observable(3);
       observable.add_operator(2.0, "X 2 Y 1 Z 0");
       observable.add_operator(-3.0, "Z 2");
       auto value = observable.get_expectation_value(&state);
       std::cout << value << std::endl;
       return 0;
   }

Example of build command:

.. code:: sh

   g++ -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lcppsim.so

Python Libraries
~~~~~~~~~~~~~~~~

You can use features by simply importing ``qulacs``.

Example of python code:

.. code:: python

   from qulacs import Observable, QuantumCircuit, QuantumState
   from qulacs.gate import Y,CNOT,merge

   state = QuantumState(3)
   state.set_Haar_random_state()

   circuit = QuantumCircuit(3)
   circuit.add_X_gate(0)
   merged_gate = merge(CNOT(0,1),Y(1))
   circuit.add_gate(merged_gate)
   circuit.add_RX_gate(1,0.5)
   circuit.update_quantum_state(state)

   observable = Observable(3)
   observable.add_operator(2.0, "X 2 Y 1 Z 0")
   observable.add_operator(-3.0, "Z 2")
   value = observable.get_expectation_value(state)
   print(value)

.. _C++ Tutorial: 
.. _Python Tutorial: 
.. _Examples: https://github.com/qulacs/quantum-circuits
.. _API document: 