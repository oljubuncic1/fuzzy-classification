from fuzzy_inference import *

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

		self.label_should_eq("rule part 0",  r[0][0],  db[0][1], [1.25, 1.5, 2, 1, 2])
		self.label_should_eq("rule part 1", r[0][1], db[1][2],[2.5, 2.75, 3, 2, 3.5, 2.5 + 0.125])
		
		self.should_eq("matching degree", matching_degree( [1.5, 2.5], r ), 0)
		self.should_eq("matching degree", matching_degree( [1.5, 3], r ), 1)

	def test_matching_degree(self):
		return

	def test_find_ranges(self):
		return

	def test_get_possible_labels(self):
		rng = (1, 2)
		label_cnt = 3

		self.should_eq("label0 middle", get_possible_labels(1, rng, label_cnt), ("0", "1"))
		self.should_eq("label0 right middle", get_possible_labels(1.25, rng, label_cnt), ("0", "1"))
		self.should_eq("label1 middle", get_possible_labels(1.5, rng, label_cnt), ("1", "2"))
		self.should_eq("label1 right middle", get_possible_labels(1.75, rng, label_cnt), ("1", "2"))
		self.should_eq("label2 middle", get_possible_labels(2, rng, label_cnt), ("1", "2"))

	def test_zinference(self):
		training_examples = [

		]

		validation_examples = [

		]

		ranges = [ (1, 3), (2, 10), (1, 4) ]
		label_cnt = 3

def test():
	test_suite = Test()
	test_suite.main()

test()