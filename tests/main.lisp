(defpackage rqga-ddp/tests/main
  (:use :cl
        :rqga-ddp
        :rove))
(in-package :rqga-ddp/tests/main)

;; NOTE: To run this test file, execute `(asdf:test-system :rqga-ddp)' in your Lisp.

(deftest test-target-1
  (testing "should (= 1 1) to be true"
    (ok (= 1 1))))
