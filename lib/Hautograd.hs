module Hautograd (
	ihautograd,
	Hautograd (..),
	OpType (..),
	sqH,
	tanhH,
	sigH,
	reluH,
	softMaxH,
) where

import qualified Data.Vector as V

data OpType = INIT | ADD | SUB | MUL | NEG | ABS | SIGNUM | SQ |
              SOFTMAX | TANH | SIG | RELU deriving (Show, Eq)

data Op = Op {
	opType :: OpType,
	gradFunction :: ()
} deriving Show

data Hautograd = Hautograd {
	value :: Float,
	ops :: Op,
	children :: V.Vector Hautograd
} deriving Show

ihautograd :: Float -> Hautograd
ihautograd val = Hautograd val (Op INIT ()) V.empty

instance Num Hautograd where 
	(+) a b = Hautograd (value a + value b) (Op ADD ()) (V.fromList [a, b])
	(-) a b = Hautograd (value a - value b) (Op SUB ()) (V.fromList [a, b])
	(*) a b = Hautograd (value a * value b) (Op MUL ()) (V.fromList [a, b])
	negate a = Hautograd (negate (value a)) (Op NEG ()) (V.singleton a)
	abs a = Hautograd (abs (value a)) (Op ABS ()) (V.singleton a)
	signum a = Hautograd (signum (value a)) (Op SIGNUM ()) (V.singleton a)
	fromInteger a = Hautograd (fromInteger a) (Op INIT ()) V.empty

sqH :: Hautograd -> Hautograd
sqH a = Hautograd (v * v) (Op SQ ()) (V.singleton a)
	where v = value a

tanhH a = Hautograd (tanh (value a)) (Op TANH ()) (V.singleton a)

sigH :: Hautograd -> Hautograd
sigH a = Hautograd (1 / (1 + exp (-v))) (Op SIG ()) (V.singleton a)
	where v = value a

reluH :: Hautograd -> Hautograd
reluH a = Hautograd (max 0 v) (Op RELU ()) (V.singleton a)
	where v = value a

softMaxH :: V.Vector Hautograd -> V.Vector Hautograd
softMaxH vec = V.map (\i -> Hautograd ((exp (value i)) / sumE) (Op SOFTMAX ()) vec) vec
  where sumE = V.sum $ V.map (exp . value) vec
