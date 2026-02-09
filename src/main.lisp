(defpackage rqga-ddp
  (:use #:cl
        #:clq
        ))
(in-package :rqga-ddp)





;; Configurations

(defconstant +METAHEURISTIC-GENE-LENGTH+ 4)
(defconstant +METAHEURISTIC-GENE-LOW-VALUE+ -5)
(defconstant +METAHEURISTIC-GENE-HIGH-VALUE+ 5)
(defconstant +METAHEURISTIC-PARENTS-MATING+ 2)
(defconstant +METAHEURISTIC-POPULATION-SIZE+ 100)
(defconstant +METAHEURISTIC-CROSSOVER-RATE+ 0.6)
(defconstant +METAHEURISTIC-MUTATION-RATE+ 0.00002)
(defconstant +METAHEURISTIC-NO-OF-GENERATIONS+ 100)



;; RQGA Configurations

(defconstant +QUBITS-FITNESS+ 8)
(defconstant +QUBITS-INDIVIDUAL+ 4)
(defconstant +POPULATION-SIZE+ (expt 2 +QUBITS-INDIVIDUAL+))
(defconstant +NO-GROVER-ITERATIONS+ (floor (sqrt (- (expt 2 +QUBITS-INDIVIDUAL+) 1))))
(defconstant +QUBITS-CARRY+ (* 2 (+ +NO-GROVER-ITERATIONS+ 1)))
(defconstant +MAX-GENERATIONS+ 8)

(defconstant +QUBITS-ANCILLA+ 2)                            






(defconstant +MAXIMUM-WEIGHT+ 10)

(defconstant +TOTAL-VALUE+ 22)

(defstruct item-configuration
  w ;;weight
  v ;;value
  )

(defparameter *items*
  (list (make-item-configuration :w 3 :v 3)
        (make-item-configuration :w 2 :v 5)
        (make-item-configuration :w 4 :v 10)
        (make-item-configuration :w 7 :v 4)))

;;Utility functions
(defun decimal-to-binary-lsb (n bits)
  "Convert decimal N to a list of BITS bits (0 or 1), least significant bit first."
  (loop for i from 0 below bits
        collect (ldb (byte 1 i) n)))


(defun fitness-function (weight value)
  (let ((quotient (floor weight +MAXIMUM-WEIGHT+)))
    (- value (* (+ +TOTAL-VALUE+ 1) quotient))))

(defun calculate-fitness-value (individual-binary)
  (let ((calculated-mass 0)
        (calculated-value 0))
    (dotimes (i (length individual-binary))
      (when (= 1 (nth i individual-binary))
        (let ((item (nth i *items*)))
          (incf calculated-mass (item-configuration-w item))
          (incf calculated-value (item-configuration-v item)))))
    (let ((fitness-value (fitness-function calculated-mass calculated-value)))
      (values (< calculated-mass +MAXIMUM-WEIGHT+) fitness-value))))
    
  

;;Quantum functions
(defun prepare-state (qc individual-register oracle-register)
  (dotimes (q +QUBITS-INDIVIDUAL+)
    (clq:hgate qc individual-register q))
  (clq:hgate qc oracle-register 0)
  )

(defun barenco-decomposition (qc ctrl-reg-0 ctrl-0 ctrl-reg-1 ctrl-1 target-reg targ)
  (clq:hgate qc target-reg targ)
  (clq:cxgate qc ctrl-reg-1 ctrl-1 target-reg targ)
  (clq:tdggate qc target-reg targ)
  (clq:cxgate qc ctrl-reg-0 ctrl-0 target-reg targ)
  (clq:tgate qc target-reg targ)
  (clq:cxgate qc ctrl-reg-1 ctrl-1 target-reg targ)
  (clq:tdggate qc target-reg targ)
  (clq:cxgate qc ctrl-reg-0 ctrl-0 target-reg targ)
  (clq:tgate qc ctrl-reg-1 ctrl-1)
  (clq:tgate qc target-reg targ)
  (clq:hgate qc target-reg targ))

  
(defun mcx-no-ancilla (qc ctrl-reg ctrl-pos target-reg targ)
  (let ((len (length ctrl-pos)))
    (cond ((= len 0)
           (clq:xgate qc target-reg targ))
          ((= len 1)
           (clq:cxgate qc ctrl-reg (car ctrl-pos) target-reg targ))
          ((= len 2)
           (barenco-decomposition qc
                                  ctrl-reg
                                  (nth 0 ctrl-pos)
                                  ctrl-reg
                                  (nth 1 ctrl-pos)
                                  target-reg
                                  targ))
          ((> len 2)
           (dotimes (i (- len 2))
             (barenco-decomposition qc
                                    ctrl-reg
                                    (nth i ctrl-pos)
                                    ctrl-reg
                                    (nth (+ i 1) ctrl-pos)
                                    target-reg
                                    targ))
           (clq:cxgate qc ctrl-reg (car (last ctrl-pos)) target-reg targ)
           (dotimes (i (- len 2))
             (barenco-decomposition qc
                                    ctrl-reg
                                    (nth i ctrl-pos)
                                    ctrl-reg
                                    (nth (+ i 1) ctrl-pos)
                                    target-reg
                                    targ))))))



(defun oracle(qc fitness-register max-register carry-register oracle-register max-value)
  (let ((max-value-binary (decimal-to-binary-lsb max-value +QUBITS-FITNESS+)))
    (dotimes (i (length max-value-binary))
      (clq:xgate qc max-register i))
    (clq:xgate qc carry-register 0)
    (adder-8 qc max-register fitness-register carry-register)
    (dotimes (i (length max-value-binary))
      (clq:xgate qc max-register i))
    (clq:xgate qc carry-register 0)
    (clq:hgate qc oracle-register 0)

    ;;TODO: Use the decomposition
    (mcx-no-ancilla qc fitness-register (list 0 1 2 3 4 5 6 7) oracle-register 0)
     
    (clq:hgate qc oracle-register 0)
    (adder-8 qc max-register fitness-register carry-register)))
  

(defun add-ufit (qc individual-register fitness-register)
  ;;just prepare the individual register
  ;;  (dotimes (individual +POPULATION-SIZE+)
  (dotimes (individual 2)
    (let ((individual-binary (decimal-to-binary-lsb individual +QUBITS-INDIVIDUAL+)))
      (dotimes (i (length individual-binary))
        (when (= 1 (nth i individual-binary))
          (clq:xgate qc individual-register i)))
      ;;calculate the fitness value and the binary representation
      (multiple-value-bind (valid fitness)
          (calculate-fitness-value individual-binary)
        (let ((fitness-binary (decimal-to-binary-lsb fitness +QUBITS-FITNESS+)))
          (dotimes (k  +QUBITS-FITNESS+)
            (when (= 1 (nth k fitness-binary))
              ;;use ancilla gates
              ;;TODO: Use the decomposition
              (mcx-no-ancilla qc individual-register (list 0 1 2 3) fitness-register k)))
          (when valid
            (mcx-no-ancilla qc individual-register (list 0 1 2 3) fitness-register +QUBITS-FITNESS+))))
      ;;reset individual
      (dotimes (i (length individual-binary))
        (when (= 1 (nth i individual-binary))
          (clq:xgate qc individual-register i))))))

  

(defun run ()
  (let* ((fitness-reg (clq:make-qregister (+ 1 +QUBITS-FITNESS+) "fr"))
         (individual-reg (clq:make-qregister +QUBITS-INDIVIDUAL+ "ir"))
         (carry-reg (clq:make-qregister +QUBITS-CARRY+ "cr"))
         (oracle-reg (clq:make-qregister 1 "or"))
         (max-reg (clq:make-qregister +QUBITS-FITNESS+ "mr"))
         (measure-reg (clq:make-cregister +QUBITS-INDIVIDUAL+ "icr"))
         (rqga-circuit (clq:make-qcircuit (list individual-reg
                                                fitness-reg
                                                carry-reg
                                                oracle-reg
                                                max-reg)
                                          (list measure-reg))))
    (prepare-state rqga-circuit individual-reg oracle-reg)
    (clq:qbarrier rqga-circuit)
    (add-ufit rqga-circuit individual-reg fitness-reg)
    (clq:qbarrier rqga-circuit)
    (oracle rqga-circuit fitness-reg max-reg carry-reg oracle-reg 10)
    (clq:save-openqasm-to-file rqga-circuit "mo.qasm")))

