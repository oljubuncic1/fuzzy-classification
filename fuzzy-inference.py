from operator import mul
from functools import reduce
import random


def triangular(center, width):
	def triangle(x):
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

	return triangle

def labels(rng, label_count):
	L = rng[1] - rng[0]
	w = 2 * L / (label_count + 1)

	return (
		[triangular( rng[0] + (i + 1) * w / 2, w ) for i in range(label_count)]
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

	# print(examples_in_class)
	# print(matching_degree(examples_in_class[0][0], rule))
	# input("Hello")

	# print(len(examples_in_class))
	# print(len(examples_out_of_class))

	in_class_md = sum(matching_degree(e[0], rule) for e in examples_in_class)
	out_of_class_md = sum(matching_degree(e[0], rule) for e in examples_out_of_class)

	if in_class_md == 0 and out_of_class_md == 0:
		return 0

	weight = (in_class_md - out_of_class_md) / (in_class_md + out_of_class_md)

	# print(weight)

	return weight

def update_weights(rb, examples):
	for r in rb:
		r[2] = weight(r, examples)

def generate_rb(examples, db):
	rb = ([rule(e, db) for e in examples])
	update_weights(rb, examples)

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

def load_csv_data(file_name, attribute_indices, class_index):
	with open(file_name) as f:
		content = f.readlines()
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


test_suite = Test()
test_suite.main()

data = load_csv_data("haberman.dat", range(3), 3)
random.shuffle(data)

validation_data_perc = 0.1
validation_examples = int(validation_data_perc * len(data))

training_data = data[:-validation_examples]
verification_data = data[validation_examples:]

label_count = 3
ranges = find_ranges(data, [0, 1, 2])
db = generate_db(ranges, label_count)

rb = generate_rb(training_data, db)

total = 0
for verification in verification_data:
	if classify(verification[0], rb) == str(verification[1]):
		total += 1

print(total / len(verification_data))
