# Modified-Crabnet
This repository contains 2 versions of modified CrabNet: 

(1) for predicting a property for each element in a formula. I this particular realisation the propety is a multiclass label.

(2) predicting multiple properties. This modified architecture has the common backbone and different output heads, one head for each property. Loss is a sum of losses from each head. Properties can be categorical or real value. In the input dataset each composition should have at least one of the properties.
