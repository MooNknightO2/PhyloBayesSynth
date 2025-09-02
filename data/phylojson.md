# PhyloJSON
A JSON format description for phylogenetic data
## Description
The format is based on [JSON file format][1] and refers to [PhyJSON format][2]
All keys are lowercase and the values are case-sensitive.
## Root element
Attributes:
Key         | Type                                   | Mandatory | Description
---         | ---                                    | :-:       | ---
format      | string                                 | Yes       | Name of the format, always "phylojson"
version     | string                                 | Yes       | Version of the format, current version is "1.0"
description | string                                 | No        | Optional description of the file contents
characters  | array of character description records | No        | Information about characters, see [Character description record](#character-description-record)
taxa        | array of taxon records                 | Yes       | Information about taxa, see [Taxon record](#taxon-record)
trees       | array of tree records                  | No        | Information about trees, see [Tree record](#tree-record)
## Character description record
Attributes:
Key         | Type             | Mandatory                 | Description
---         | ---              | :-:                       | ---
id          | string           | Yes                       | Unique identifier of the character
description | string           | No                        | Optional description of the file contents
type        | string           | Yes                       | Type of data: "dna", "rna", "protein", "nucleotide", "standard" or "continuous"
aligned     | boolean          | No                        | True if the character sequences are aligned, false otherwise
missing     | string           | No                        | The missing symbol (defaults to "?")
gap         | string           | No                        | The gap symbol (defaults to "-")
symbols     | array of strings | Yes if type is "standard" | A list of allowed symbols. Symbols are ingored for any other type than "standard". The symbols must not contain the missing symbol, the gap symbol, parantheses and braces, quote marks (both single and double) and comma.

Example:
```
"characters": [
		{"id": "dna", "type": "dna", "aligned": true},
		{"id": "state", description: "BiSSE state", "type": "standard", symbols: ["0", "1"]},
	]
```

## Taxon record
Attributes:
Key        | Type              | Mandatory | Description
---        | ---               | :-:       | ---
id         | string or integer | Yes       | Unique identifier of the taxon (used to refer to the taxon)
name       | string            | No        | Unique name of the taxon
characters | object            | No        | An object with the keys referring to the identifier of the character (see [Character description record](#character-description-record) above). Each value follows the description [below](#character-data).

Example:
```
"taxa": [
		{"id": 1, "name": "Taxon 1", "characters": {"dna": "cgggtccctctggtgactggct?gatggac", "state": "0"}},
		{"id": 2, "name": "Taxon 2", "characters": {"dna": "ctctctattgatgtcacggcgaatgtcggg", "state": "1"}}
	]
```

## Character data

Symbols are separated by comma. The comma can be dropped if all symbols are single characters. Multiple states might be specified using the {} and () notation as in the NEXUS format. Array may also be used.

Example: Character data in the following two examples are equivalent:

```
"actg"
"a,c,t,g"
["a", "c", "t", "g"]
```

```
"a-c{ag}?"
"a,-,c,{a,g},?"
["a", "-", "c", "{ag}", "?"]
["a", "-", "c", ["a", "g"], "?"]
```

For the continuous characters, the data is specified as an array of numbers.

## Tree record

Key    | Type                | Mandatory | Description
---    | ---                 | :-:       | ---
name   | string              | No        | Unique name of the tree
rooted | boolean             | No        | True (default) if the tree is rooted, false otherwise.
root   | object of type node | Yes       | For a rooted tree: the root node. For an unrooted tree: the root of an arbitrary rooting

Example:
```
"trees": [
		{
			"name": "Some tree",
			"rooted": true,
			"root": {
				"branch_length": 0.7,
				"children": [
					{"taxon": 1, "branch_length": 1.2},
					{"taxon": 2, "branch_length": 1.2}
				]
			}
		}
	]
```

## Node record

Key           | Type                  | Mandatory | Description
---           | ---                   | :-:       | ---
taxon         | string                | No        | The taxon ID
branch_length | number                | No        | The length of the branch between the node's parent and the node (the stalk length for the root)
children      | array of node records | No        | Children of the node






[1]: https://json.org/
[2]: https://github.com/kudlicka/nexus2phyjson/blob/master/doc/phyjson_format_description.md