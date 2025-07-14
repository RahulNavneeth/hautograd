module Main where

import qualified Data.Vector as V
import Hautograd

main :: IO ()
main = print (reluH (ihautograd 34.0 + ihautograd 35.0))
