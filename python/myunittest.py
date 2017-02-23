from inspect import ismethod



class Test():

    def __init__(self):
        return

    def should_eq(self, name, val1, val2):
        BEGIN_PASS = '\033[92m'
        BEGIN_FAIL = '\033[91m'
        END = '\033[0m'

        if val1 == val2:
            print(
                "\t" + BEGIN_PASS + "{" + name + "}" + END + "\t--> " + 
                str(val1) + " = " + str(val2)
            )
        else:
            print(
                "\t" + BEGIN_FAIL + "{" + name + "}" + END + "\t--> " + 
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
