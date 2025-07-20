module Hautograd (
	ihautograd,
	Hautograd (..),
	Op (..),
	sqH,
	tanhH,
	sigH,
	reluH,
	softMaxH,
) where

import qualified Data.Vector as V

data Op = INIT | ADD | SUB | MUL | NEG | ABS | SIGNUM | SQ |
              SOFTMAX | TANH | SIG | RELU deriving (Show, Eq)

type Grad = Float; 

data Hautograd = Hautograd {
	value :: Float,
	ops :: Op,
	grad :: Grad,
	children :: V.Vector Hautograd
} deriving Show

ihautograd :: Float -> Hautograd
ihautograd val = Hautograd val INIT 0.01 V.empty

instance Num Hautograd where 
	(+) a b = Hautograd (value a + value b) ADD (V.fromList [a, b])
	(-) a b = Hautograd (value a - value b) SUB (V.fromList [a, b])
	(*) a b = Hautograd (value a * value b) MUL (V.fromList [a, b])
	negate a = Hautograd (negate (value a)) NEG (V.singleton a)
	abs a = Hautograd (abs (value a)) ABS (V.singleton a)
	signum a = Hautograd (signum (value a)) SIGNUM (V.singleton a)
	fromInteger a = Hautograd (fromInteger a) INIT V.empty

sqH :: Hautograd -> Hautograd
sqH a = Hautograd (v * v) SQ (V.singleton a)
	where v = value a

tanhH a = Hautograd (tanh (value a)) TANH (V.singleton a)

sigH :: Hautograd -> Hautograd
sigH a = Hautograd (1 / (1 + exp (-v))) SIG (V.singleton a)
	where v = value a

reluH :: Hautograd -> Hautograd
reluH a = Hautograd (max 0 v) RELU (V.singleton a)
	where v = value a

softMaxH :: V.Vector Hautograd -> V.Vector Hautograd
softMaxH vec = V.map (\i -> Hautograd ((exp (value i)) / sumE) SOFTMAX vec) vec
  where sumE = V.sum $ V.map (exp . value) vec
