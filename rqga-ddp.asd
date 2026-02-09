(defsystem "rqga-ddp"
  :version "0.0.1"
  :author ""
  :license ""
  :depends-on ("clq")
  :components ((:module "src"
                :serial t  ; Ensures files are loaded in order
                :components
                ((:file "adder")   ;; load this first (defines package)
                 (:file "main"))))  ;; load this second
  :description ""
  :in-order-to ((test-op (test-op "rqga-ddp/tests"))))

(defsystem "rqga-ddp/tests"
  :author ""
  :license ""
  :depends-on ("rqga-ddp"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for rqga-ddp"
  :perform (test-op (op c) (symbol-call :rove :run c)))
