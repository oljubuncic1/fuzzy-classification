from operator import mul
from functools import reduce
import random
import logging
from collections import defaultdict
from functools import partial
import math
import ann

logging.basicConfig(level=logging.DEBUG)

def triangle(center, width, name, x):
	r = width / 2
	k = 1 / r
	
	left = center - r
	right = center + r

	if x == 'name':
		return name #str(left) + " " + str(right)
	else:
		x = float(x)

		# if r**2 - (x - center) ** 2 <= 0:
		# 	return 0
		# else:
		# 	return (1 / r) * float( (r**2 - (x - center) ** 2) ** 0.5 )

		if left <= x <= center:
			return k * (x - left) + 0
		elif center <= x <= right:
			return -k * (x - center) + 1
		else:
			return 0

def triangular(center, width, name='x'):
	return partial(triangle, center, width, name)

def labels(rng, label_count):
	L = rng[1] - rng[0]
	w = L / (label_count - 2)

	if w == 0:
		w = 1 # hacky, because division by zero

	return (
		[triangular( rng[0] + i * w / 2.0, w, str(i) ) for i in range(label_count)]
	)

def generate_db(ranges, label_count):
	return [labels(r, label_count) for r in ranges]

def classification(example):
	return example[1]

def input_data(example):
	return example[0]

def matching_degree(example, x):
	l = [  x[0][i](float(example[i])) for i in range(len(example)) ]
	return min(l)
	
	dgr = reduce(mul, l, 1)
	return dgr

	dgr = 1
	for i in range(len(example)):
		curr = x[0][i](float(example[i]))
		if curr == 0:
			return 0
		else:
			dgr = dgr * curr

	return dgr

def rule(example, db):
	data = input_data(example)
	rw = 0.1
	rule = [[], classification(example), rw]

	for i in range(len(db)):
		max_label = max(db[i], key=lambda x: x(data[i]))
		# print(max_label(data[i]))
		rule[0].append( max_label )

	rule[2] = matching_degree(data, rule)
	# logging.debug("Matching degree to rule " + str(rule[2]))

	return rule

def weight(rule, examples):
	examples_in_class = [x for x in examples if x[1] == rule[1]]
	examples_out_of_class = [x for x in examples if x[1] != rule[1]]

	in_class_md = sum(matching_degree(e[0], rule) for e in examples_in_class)
	out_of_class_md = sum(matching_degree(e[0], rule) for e in examples_out_of_class)

	if in_class_md == 0 and out_of_class_md == 0:
		return 0

	weight = (in_class_md - out_of_class_md) / (in_class_md + out_of_class_md)

	return weight

def update_weights(rb, examples):
	for r in rb:
		r[2] = weight(r, examples)
	return rb

def update_weight(r, examples):
	r[2] = weight(r, examples)
	return r

def rule_desc(rule):
	rule_desc_str = ""

	for f in rule[0]:
		rule_desc_str += f('name')
	rule_desc_str += rule[1]

	return rule_desc_str

def remove_duplicates(rb):
	rule_map = defaultdict(list)
	for r in rb:
		rule_map[rule_desc(r)].append(r)

	best_rules = []
	
	for k in rule_map:
		best_rules.append(max(rule_map[k], key=lambda x: x[2]))

	rb = best_rules

from itertools import product
from multiprocessing import Pool

def generate_rb(examples, db):
	rb = None
	with Pool(processes=4) as pool:
		logging.debug("Creating rules...")
		rb = pool.starmap(rule, [ (e, db) for e in examples ])
		# remove really same rules
		logging.debug("Updating weights...")
		rb = pool.starmap(update_weight, [ (r, list(examples)) for r in rb ] )

	logging.debug("Removing duplicates...")
	remove_duplicates(rb)

	rb_map = {}
	for r in rb:
		rule_str = ""
		for i in range(len(r[0])):
			rule_str += r[0][i]('name')
		rb_map[rule_str] = r

	# return rb
	return rb_map

def generate_possible_rules(example, ranges, label_cnt, lvl = 0, curr = ""):
	# example is just list of atttributes here
	if lvl == len(ranges):
		return [curr]
	else:
		L = ranges[lvl][1] - ranges[lvl][0]
		W = L / (label_cnt - 1) 
		
		first = None
		second = None

		val = float(example[lvl])
		if val != ranges[lvl][0] and val != ranges[lvl][1]:
			val = val + 0.1

		ind = None
		if W == 0:
			ind = 0
		else:
			ind = int( ( val - ranges[lvl][0] ) / (W / 2) )

		if ind == 0:
			first = "0"
			second = "1"
		elif ind == label_cnt + 1:
			first = str(label_cnt - 1)
			second = str(label_cnt - 2)
		elif ind % 2 == 0:
			first = str(int(ind / 2))
			second = str(int(first) + 1)
		else:
			second = str( int( (ind + 1) / 2) )
			first = str(int(second) - 1)

		if val <= ranges[lvl][0] + L / 4:
			first = "0"
			second = "1"
		elif ranges[lvl][0] + L / 4 <= val <= ranges[lvl][0] + L / 2:
			first = "0"
			second = "1"
		elif ranges[lvl][0] + L / 2 <= val <= ranges[lvl][0] + L / 2 + L / 4:
			first = "1"
			second = "2"
		else:
			first = "1"
			second = "2"

		# first = ( "0" * int(math.log(label_cnt, 10) - len(first) + 1) ) + first
		# second = ( "0" * int(math.log(label_cnt, 10) - len(first) + 1) ) + second

		return generate_possible_rules(example, ranges, label_cnt, lvl + 1, curr + first) + \
			generate_possible_rules(example, ranges, label_cnt, lvl + 1, curr + second)

def classify(example, rb, ranges, label_cnt):
	# example is just list of atttributes here
	possible_rules = generate_possible_rules(example, ranges, label_cnt)
	
	max_degree = 0
	max_classification = "0"
	for r in possible_rules:
		if r not in rb:
			continue
		curr_md = matching_degree(example, rb[r]) * rb[r][2]
		if curr_md > max_degree:
			max_degree = curr_md
			max_classification = rb[r][1]

	print(max_degree)

	return max_classification

	max_rule = rb[0]
	max_rule_dgr = matching_degree(example, max_rule)
	
	for r in rb:
		curr_matching_degree = matching_degree(example, r)
		if(curr_matching_degree > max_rule_dgr):
			max_rule_dgr = curr_matching_degree
			max_rule = r

	return max_rule[1]

	max_rule = max(rb, key=lambda x: matching_degree(example, x) * x[2])
	return max_rule[1]

def example_from_line(line, attribute_indices, class_index, symbol):
	parts = line.split(symbol)

	classification = parts[ class_index ]
	parts = [ parts[i].strip() for i in attribute_indices ]

	example = [
		parts,
		classification
	]

	return example

def load_csv_data(file_name, attribute_indices, class_index, line_cnt, symbol):
	with open(file_name) as f:
		content = head = [next(f) for x in range(line_cnt)]
		lines = [x.strip() for x in content]
		data = [
			example_from_line(l,attribute_indices, class_index, symbol)
			for l in lines
		]

		return data

def find_ranges(examples, indices, discrete_indices = []):
	ranges = []
	for i in indices:
		ranges.append((
			min([float(e[0][i]) for e in examples]),
			max([float(e[0][i]) for e in examples])
		))

	return ranges


from inspect import ismethod

class Test:
	def should_eq(self, name, val1, val2):
		BEGIN_PASS = '\033[92m'
		BEGIN_FAIL = '\033[91m'
		END = '\033[0m'

		if val1 == val2:
			print(
				"\t{" + name + "}" + BEGIN_PASS + " pass --> " + 
				str(val1) + " = " + str(val2) + END
			)
		else:
			print(
				"\t{" + name + "}" + BEGIN_FAIL + " fail --> " + 
				str(val1) + " != " + str(val2) + END
			)

	def main(self):
		for name in dir(self):
			attribute = getattr(self, name)
			if ismethod(attribute) and attribute.__name__.startswith('test'):
				print(attribute.__name__)
				attribute()

		print("")
		print("")
		print("")

	def test_triangular(self):
		f = triangular(1, 2)

		self.should_eq( "leftmost", f(0), 0 )
		self.should_eq( "middle left", f(0.5), 0.5 )
		self.should_eq( "center", f(1), 1 )
		self.should_eq( "middle right", f(1.5), 0.5 )
		self.should_eq( "rightmost", f(2), 0 )
		self.should_eq( "other 1", f(-5), 0 )
		self.should_eq( "other 2", f(5), 0 )

	def test_labels_whole(self):
		lbls = labels( (0, 4), 3 )

		self.should_eq( "label count", len(lbls), 3 )

		f = lbls[0]
		self.should_eq( "label0 left", f(0), 1 )
		self.should_eq( "label0 middle left", f(2), 0 )

		f = lbls[1]
		self.should_eq( "label1 left", f(0), 0 )
		self.should_eq( "label1 middle", f(2), 1 )
		self.should_eq( "label1 right", f(4), 0 )

	def test_labels_fractions(self):
		lbls = labels( (1, 2), 3 )

		self.should_eq( "label count", len(lbls), 3 )

		f = lbls[0]
		self.should_eq( "label0 center", f(1), 1 )
		self.should_eq( "label0 right", f(1.5), 0 )
		for i in [-2, -1, -0.5, 0]:
			self.should_eq( "label0 left out at " + str(i), f(i), 0 )
		for i in [2.5, 3,  3.5]:
			self.should_eq( "label0 left out at " + str(i), f(i), 0 )

		f = lbls[1]
		self.should_eq( "label1 left", f(1), 0 )
		self.should_eq( "label1 right", f(1.5), 1 )
		self.should_eq( "label1 right", f(2), 0 )
		for i in [-2, -1, -0.5, 0]:
			self.should_eq( "label1 left out at " + str(i), f(i), 0 )
		for i in [2.5, 3,  3.5]:
			self.should_eq( "label1 left out at " + str(i), f(i), 0 )

		f = lbls[2]
		self.should_eq( "label1 left", f(1.5), 0 )
		self.should_eq( "label1 right", f(2), 1 )
		for i in [-2, -1, -0.5, 0]:
			self.should_eq( "label1 left out at " + str(i), f(i), 0 )
		for i in [2.5, 3,  3.5]:
			self.should_eq( "label1 left out at " + str(i), f(i), 0 )

	def test_create_db(self):
		ranges = [ (1, 2), (2, 3) ]
		label_count = 5
		db = generate_db(ranges, label_count)

		self.should_eq( "two inputs", len(db), 2 )
		self.should_eq( "labels per input", len( db[0] ), label_count )

	def label_should_eq(self, name, label1, label2, points):
		print("\t" + name)
		for point in points:
			self.should_eq( "point " + str(point), label1(point), label2(point) )

	def test_rule(self):
		ranges = [ (1, 2), (2, 3) ]
		label_cnt = 3
		db = generate_db(ranges, label_cnt)

		example = ( [1.5, 3], 2 )
		r = rule(example, db)

		self.should_eq( "rule is triplet", len(r), 3 )
		self.should_eq( "rule input number", len(r[0]), len(example[0]) )
		self.should_eq( "classification", r[1], example[1] )

		self.label_should_eq(
			"rule part 0", 
			r[0][0], 
			db[0][1],
			[1.25, 1.5, 2, 1, 2]
		)
		
		self.label_should_eq(
			"rule part 1", 
			r[0][1], 
			db[1][2],
			[2.5, 2.75, 3, 2, 3.5, 2.5 + 0.125]
		)

		self.should_eq(
			"matching degree", 
			matching_degree( [1.5, 2.5], r ), 
			0
		)

		self.should_eq(
			"matching degree", 
			matching_degree( [1.5, 3], r ), 
			1
		)

	def test_matching_degree(self):
		return

	def test_weight(self):
		ranges = [ (1, 2), (2, 3) ]
		label_cnt = 3
		db = generate_db(ranges, label_cnt)

		examples = [ [ [1.5, 2.5 - 1/8], 2 ] ]
		r = rule(examples[0], db)

		w = weight(r, examples)

		self.should_eq("weight", w, 1.0)

	def test_find_ranges(self):
		return

	def test_rule_desc(self):
		return

	def test_remove_duplicates(self):
		return

	def test_generate_possible_rules(self):
		# example, ranges, label_cnt, lvl = 0, curr = ""
		ranges = [ (1, 2), (1.5, 2.5), (-3, 3) ]
		label_cnt = 3
		
		print(ranges)

		example = [ 1, 1.5, -3 ]
		print(example)
		print( generate_possible_rules(example, ranges, label_cnt) )

		example = [ 1, 2.1, -3 ]
		print(example)
		print( generate_possible_rules(example, ranges, label_cnt) )

		example = [ 1, 2.3, -3 ]
		print(example)
		print( generate_possible_rules(example, ranges, label_cnt) )

def test():
	test_suite = Test()
	test_suite.main()

def print_rb(rb, rw_pos = 3):
	for rule in rb:
		rule_str = ''
		i = 0
		for f in rule[0]:
			rule_str +=  'x' + str(i) + ' is ' + f('name') + '   '
			i += 1
		rule_str += " class is " + str(rule[1]) + " with rw " + str(round(rule[2], rw_pos))
		print(rule_str)

def load_metadata():
	metadata = {}

	metadata[0] = 'banana'
	metadata[1] = 'wine'
	metadata[2] = 'wine'
	metadata[3] = 'banana'
	metadata[4] = 'wine'
	metadata[5] = 'banana'
	metadata[6] = 'wine'
	metadata[7] = 'banana'
	metadata[8] = 'wine'
	metadata[9] = 'banana'
	metadata[10] = 'banana'
	metadata[11] = 'banana'
	metadata[12] = 'wine'
	metadata[13] = 'banana'
	metadata[14] = 'wine'
	metadata[15] = 'banana'
	metadata[16] = 'banana'
	metadata[17] = 'banana'
	metadata[18] = 'banana'
	metadata[19] = 'wine'
	metadata[20] = 'wine'
	metadata[21] = 'banana'
	metadata[22] = 'banana'
	metadata[23] = 'wine'
	metadata[24] = 'wine'
	metadata[25] = 'wine'
	metadata[26] = 'wine'
	metadata[27] = 'banana'
	metadata[28] = 'wine'
	metadata[29] = 'wine'
	metadata[30] = 'banana'
	metadata[31] = 'wine'
	metadata[32] = 'wine'
	metadata[33] = 'banana'
	metadata[34] = 'banana'
	metadata[35] = 'banana'
	metadata[36] = 'wine'
	metadata[37] = 'wine'
	metadata[38] = 'wine'
	metadata[39] = 'banana'
	metadata[40] = 'banana'
	metadata[41] = 'wine'
	metadata[42] = 'wine'
	metadata[43] = 'banana'
	metadata[44] = 'wine'
	metadata[45] = 'wine'
	metadata[46] = 'wine'
	metadata[47] = 'banana'
	metadata[48] = 'banana'
	metadata[49] = 'wine'
	metadata[50] = 'wine'
	metadata[51] = 'banana'
	metadata[52] = 'banana'
	metadata[53] = 'wine'
	metadata[54] = 'wine'
	metadata[55] = 'banana'
	metadata[56] = 'wine'
	metadata[57] = 'banana'
	metadata[58] = 'wine'
	metadata[59] = 'banana'
	metadata[60] = 'banana'
	metadata[61] = 'wine'
	metadata[62] = 'banana'
	metadata[63] = 'banana'
	metadata[64] = 'wine'
	metadata[65] = 'banana'
	metadata[66] = 'wine'
	metadata[67] = 'wine'
	metadata[68] = 'wine'
	metadata[69] = 'background'
	metadata[70] = 'background'
	metadata[71] = 'background'
	metadata[72] = 'background'
	metadata[73] = 'background'
	metadata[74] = 'background'
	metadata[75] = 'background'
	metadata[76] = 'background'
	metadata[77] = 'background'
	metadata[78] = 'background'
	metadata[79] = 'background'
	metadata[80] = 'background'
	metadata[81] = 'background'
	metadata[82] = 'background'
	metadata[83] = 'background'
	metadata[84] = 'background'
	metadata[85] = 'background'
	metadata[86] = 'background'
	metadata[87] = 'background'
	metadata[88] = 'background'
	metadata[89] = 'background'
	metadata[90] = 'background'
	metadata[91] = 'background'
	metadata[92] = 'background'
	metadata[93] = 'background'
	metadata[94] = 'background'
	metadata[95] = 'background'
	metadata[96] = 'background'
	metadata[97] = 'background'
	metadata[98] = 'background'
	metadata[99] = 'background'

	return metadata

def main():
	logging.info("Loading data...")
	# cols = [0, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 
		# 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
	cols = range(10)
	data = load_csv_data(
		"../data/poker-hand-testing.data", 
		cols,
		10,
		1000000,
		','
	)
	logging.info("Data loaded")
	random.shuffle(data)
	data = data[0:3000]

	for d in data:
		if d[1] == '4':
			d[1] = '1'
		elif d[1] != '1':
			d[1] = '0'

	# metadata = load_metadata()

	# for item in data:
	# 	item[1] = metadata[int(item[1])]

	validation_data_perc = 0.1
	validation_examples = int(validation_data_perc * len(data))

	training_data = data[:-validation_examples]
	verification_data = data[-validation_examples:]

	logging.info("Generating db...")
	label_count = 3
	ranges = find_ranges(data, range(len(cols)))

	# for d in data:
	# 	for i in range(len(d[0])):
	# 		d[0][i] = int(d[0][i]) - ranges[i][0]
	# 		d[0][i] = d[0][i] * 100 / ( ranges[i][1] - ranges[i][0] )
	# 		d[0][i] = str(d[0][i])

	# print(data)

	db = generate_db( ranges , label_count)

	logging.info("Generating rb...")
	rb = generate_rb(training_data, db)

	logging.info("Classifying...")

	with Pool(processes=4) as pool:
		classifications = pool.starmap(classify, [ (v[0], rb, list(ranges), label_count) for v in verification_data])

	total = 0
	for i in range(len(verification_data)):
		verification = verification_data[i]
		if classifications[i] == str(verification[1]):
			total += 1

	print( "Accuracy% " + str(100 * total / len(verification_data)) )

	for c in set( [ x[1] for x in verification_data ] ):
		print(
			c + " " + str( len( [x for x in verification_data if x[1] == c] ) )
		)

main()
