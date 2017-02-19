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

def test():
    return

if __name__ == 'main':
    test()
