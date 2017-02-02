from operator import mul
from functools import reduce
import random
import logging
from collections import defaultdict
from functools import partial

logging.basicConfig(level=logging.DEBUG)

def triangle(center, width, name, x):
	if x == 'name':
		return name
	else:
		x = float(x)

		r = width / 2
		k = 1 / r
		
		left = center - r
		right = center + r

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
	w = 2 * L / (label_count + 1)

	if w == 0:
		w = 1 # hacky

	return (
		[triangular( rng[0] + (i + 1) * w / 2, w, str(i) ) for i in range(label_count)]
	)

def generate_db(ranges, label_count):
	return  [labels(r, label_count) for r in ranges]

def classification(example):
	return example[1]

def input_data(example):
	return example[0]

def matching_degree(example, x):
	l = [ x[0][i](example[i]) for i in range(len(example)) ]

	return reduce(mul, l, 1)

def rule(example, db):
	data = input_data(example)
	rw = 0.1
	rule = [[], classification(example), rw]

	for i in range(len(db)):
		rule[0].append( max(db[i], key=lambda x: x(data[i])) )

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

def update_weight(r, examples):
	r[2] = weight(r, examples)
	return r

def rule_desc(rule):
	rule_desc_str = ""

	for f in rule[0]:
		rule_desc_str += f('name')

	return rule_desc_str

def remove_duplicates(rb):
	rule_map = defaultdict(list)
	for r in rb:
		rule_map[rule_desc(r)].append(r)
		rule_map[rule_desc(r)] = [ max(rule_map[rule_desc(r)], key=lambda x: x[2]) ]

	return [rule_map[k][0] for k in rule_map]

	best_rules = []
	
	for k in rule_map:
		best_rules.append(max(rule_map[k], key=lambda x: x[2]))

	rb = best_rules

from multiprocessing import Pool

from itertools import product
from multiprocessing import Pool

def generate_rb(examples, db):
	rb = None
	with Pool(processes=4) as pool:
		rb = pool.starmap(rule, [ (e, db) for e in examples ])
		rb = pool.starmap(update_weight, [ (r, list(examples)) for r in rb ] )
	
	remove_duplicates(rb)

	return rb

def classify(example, rb):
	max_rule = max(rb, key=lambda x: matching_degree(example, x))

	return max_rule[1]

def example_from_line(line, attribute_indices, class_index):
	parts = line.split(',')

	classification = parts[ class_index ]
	parts = [ parts[i].strip() for i in attribute_indices ]

	example = (
		parts,
		classification
	)

	return example

def load_csv_data(file_name, attribute_indices, class_index, line_cnt):
	with open(file_name) as f:
		content = head = [next(f) for x in range(line_cnt)]
		lines = [x.strip() for x in content]
		data = [
			example_from_line(l,attribute_indices, class_index)
			for l in lines
		]

		return data

def find_ranges(examples, indices):
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
		self.should_eq( "label0 left", f(0), 0 )
		self.should_eq( "label0 middle left", f(0.5), 0.5 )
		self.should_eq( "label0 center", f(1), 1 )
		self.should_eq( "label0 middle right", f(1.5), 0.5 )
		self.should_eq( "label0 right", f(2), 0 )

		f = lbls[1]
		self.should_eq( "label1 left", f(1), 0 )
		self.should_eq( "label1 middle left", f(1.5), 0.5 )
		self.should_eq( "label1 center", f(2), 1 )
		self.should_eq( "label1 middle right", f(2.5), 0.5 )
		self.should_eq( "label1 right", f(3), 0 )

	def test_labels_fractions(self):
		lbls = labels( (1, 2), 3 )

		self.should_eq( "label count", len(lbls), 3 )

		f = lbls[0]
		self.should_eq( "label0 left", f(1), 0 )
		self.should_eq( "label0 center", f(1.25), 1 )
		self.should_eq( "label0 right", f(1.5), 0 )

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

		example = ( [1.5, 2.75], 2 )
		r = rule(example, db)

		self.should_eq( "rule is triplet", len(r), 3 )
		self.should_eq( "rule input number", len(r[0]), len(example[0]) )
		self.should_eq( "classification", r[1], example[1] )

		self.label_should_eq(
			"rule part 0", 
			r[0][0], 
			triangular(1.5, 0.5),
			[1.25, 1.5, 2, 1, 2]
		)
		
		self.label_should_eq(
			"rule part 1", 
			r[0][1], 
			triangular(2.75, 0.5),
			[2.5, 2.75, 3, 2, 3.5, 2.5 + 0.125]
		)

		self.should_eq(
			"matching degree", 
			matching_degree( [1.5, 2.5 + 0.125], r ), 
			0.5
		)

		self.should_eq(
			"matching degree", 
			matching_degree( [1.5 + 0.125, 2.5 + 0.125], r ), 
			0.25
		)

	def test_weight(self):
		return

	def test_find_ranges(self):
		return

	def test_rule_desc(self):
		return

	def test_remove_duplicates(self):
		return


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

logging.info("Loading data...")
cols = [0, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 
	30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
data = load_csv_data(
	"kddcupfull.data", 
	cols,
	41,
	2000
)
logging.info("Data loaded")
random.shuffle(data)
data = data[-2000:]

validation_data_perc = 0.1
validation_examples = int(validation_data_perc * len(data))

training_data = data[:-validation_examples]
verification_data = data[validation_examples:]

logging.info("Generating db...")
label_count = 3
ranges = find_ranges(data, range(len(cols)))
db = generate_db(ranges, label_count)

logging.info("Generating rb...")
rb = generate_rb(training_data, db)

logging.info("Classifying...")
total = 0

with Pool(processes=4) as pool:
	classifications = pool.starmap(classify, [ (v, list(rb)) for v in verification_data])

print(classifications)

# for verification in verification_data:
# 	if classify(verification[0], rb) == str(verification[1]):
# 		total += 1

# print( "Accuracy% " + str(100 * total / len(verification_data)) )
