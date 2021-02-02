# Ogre interface

This code is a fast-screening computational tool for structure prediction of organic/inorganic epitaxial interfaces. 

The inputs include:
- ogre_interfaces.ini: Includes setting for lattice matching and structure generation
- The bulk structure of substrate and film

## Installation
* `python ./setup.py develop`

## Basic Command
Create organic interfaces:
'python main.py --path test/organic/'

Create organic Interfaces with repaired molecules:
'python main.py --path test/organic/ --repair True'

Create inorganic interfaces:
'python main.py --path test/inorganic/'
