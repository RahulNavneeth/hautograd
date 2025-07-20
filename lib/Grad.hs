module Grad (
add', sub', mul', neg', abs', signum',
sq', tanh', sig', relu', softmax',
) where

import Hautograd

updateGrad :: Hautograd -> Grad  -> Hautograd
updateGrad hg gradUpdate = hg { grad = (grad hg) + gradUpdate }

add' :: Hautograd -> Hautograd -> Hautograd -> (Hautograd, Hautograd)
add' parent hga hgb = (updateGrad hga (grad parent), updateGrad hgb (grad parent))

sub' :: Hautograd -> Hautograd -> Hautograd -> (Hautograd, Hautograd)
sub' parent hga hgb = (updateGrad hga (grad parent), updateGrad hgb (negate (grad parent)))

mul' :: Hautograd -> Hautograd -> Hautograd -> (Hautograd, Hautograd)
mul' parent hga hgb = (updateGrad hga ((value hgb) * grad parent), updateGrad hgb ((value gha) * (grad parent)))

neg' :: Hautograd -> Hautograd -> Hautograd
neg' parent hg = updateGrad hg (negate (grad parent))

abs' :: Hautograd -> Hautograd -> Hautograd
abs' parent hg
    | (value hg) > 0 = (updateGrad hg (grad parent))
    | (value hg) < 0 = (updateGrad hg (grad parent))
    | otherwise = (updateGrad hg 0.0)

signum' :: Hautograd -> Hautograd -> Hautograd
signum' parent hg
    | (value hg) > 0 = (updateGrad hg (grad parent))
    | (value hg) < 0 = (updateGrad hg (grad parent))
    | otherwise = (updateGrad hg 0.0)

sq' :: Hautograd -> Hautograd -> Hautograd
sq' parent hg = updateGrad hg ((2.0 * (value hg)) * (grad parent)) 

tanh' :: Hautograd -> Hautograd -> Hautograd
tanh' parent hg = updateGrad hg ((1 - (value hg) ^ 2) * grad parent)

sig' :: Hautograd -> Hautograd -> Hautograd
sig' parent hg = updateGrad hg ((value hg) * (1 - value hg) * grad parent)

relu' :: Hautograd -> Hautograd -> Hautograd
relu' parent hg = updateGrad hg (if value hg > 0 then grad parent else 0)

softmax' :: V.Vector Hautograd -> V.Vector Hautograd -> V.Vector Hautograd
softmax' parents grads = V.imap update grads
  where
    update i hg_i = 
      let yi = value hg_i
          grad_i = V.sum $ V.imap (\j hg_j ->
                      let yj = value hg_j
                          gij = if i == j then yi * (1 - yi) else -yi * yj
                      in gij * grad (parents V.! j)
                    ) grads
      in updateGrad hg_i grad_i
