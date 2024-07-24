from config import *
from helper import *
from num import NUM
from sym import SYM
from data import DATA
from main import bayes

class Tests:
    def test_num_small_zero_div(self):
        num = NUM()
        num.n = 1
        the = {'cohen': 0.35, 'file': 'w3/data/diabetes.csv', 'help': False, 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'test': 'all'}
        assert num.small(the) == 0

    def test_num_small_nonzero_div(self):
        num = NUM()
        num.n = 3
        num.m2 = 5
        the = {'cohen': 1, 'file': 'w3/data/diabetes.csv', 'help': False, 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'test': 'all'}
        assert num.small(the) == (5 / 2)**0.5

    def test_num_small_cohen_factor(self):
        num = NUM()
        num.n = 3
        num.m2 = 5
        the = {'cohen': 2, 'file': 'w3/data/diabetes.csv', 'help': False, 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'test': 'all'}
        expected_result = the['cohen'] * (5 / 2)**0.5
        assert num.small(the) == expected_result
        
    def test_add_sym(self):
        sym = SYM("origin",5)
        sym.add("1")
        assert sym.n == 1
        assert sym.has == {"1": 1}
        assert sym.mode == "1"
        assert sym.most == 1

        # testing by adding a row with same origin value
        sym.add("1")
        assert sym.n == 2
        assert sym.has == {"1": 2}
        assert sym.mode == "1"
        assert sym.most == 2

    def test_mid_sym(self):
        sym = SYM("origin",5)
        sym.add("1")
        sym.add("2")
        assert sym.mid() == "1"

    def test_div_sym(self):
        sym = SYM("origin",5)
        sym.add("1")
        sym.add("1")

        assert sym.div() == 0

    def test_small_sym(self):
        sym = SYM("origin",5)
        sym.add("1")

        assert sym.small(0) == 0

    def test_coerce(self):
        assert coerce("true") == True
        assert coerce("93") == 93
        assert coerce("46.18") == 46.18
        assert coerce("  coerce  ") == "coerce"
        assert coerce("nil") == None
    
    def test_cells(self):
        input = "homework, 44, false, true, 22.94"
        output = cells(input)
        assert output == ["homework", 44, False, True, 22.94]
    
    def test_roundoff(self):
        assert round(42.5421, 2) == 42.54
        assert round(70.853, 2) == 70.85

    def test_mid_num(self):
        test_num = NUM()
        test_num.add(10)
        test_num.add(12)
        assert test_num.mid() == 11

    def test_add_num(self):
        num_obj = NUM()
        num_obj.add(50)
        assert num_obj.n == 1
        assert num_obj.mu == 50
        assert num_obj.m2 == 0.0
        assert num_obj.lo == 50
        assert num_obj.hi == 50
        num_obj.add(165)
        assert num_obj.n == 2
        assert num_obj.mu == 107.5
        assert num_obj.m2 == 6612.5
        assert num_obj.lo == 50
        assert num_obj.hi == 165
        num_obj.add(127)
        assert num_obj.n == 3
        assert num_obj.mu == 114.0
        assert num_obj.m2 == 6866.0
        assert num_obj.lo == 50
        assert num_obj.hi == 165
        num_obj.add(16)
        assert num_obj.n == 4
        assert num_obj.mu == 89.5
        assert num_obj.m2 == 14069.0
        assert num_obj.lo == 16
        assert num_obj.hi == 165

    def test_div_num(self):
        num_obj = NUM()
        num_obj.add(5)
        assert num_obj.div() == 0
        num_obj.add(10)
        assert num_obj.div() == (12.5)**0.5

    def test_sym_mode(self):
        test_sym = SYM()
        for x in [1, 1, 1, 1, 2, 2, 3]:
            test_sym.add(x)
        mode = test_sym.mid()
        assert mode == 1
    
    def test_sym_like(self):
        sym_instance = SYM()
        sym_instance.add("A")
        sym_instance.add("B")
        sym_instance.add("A")
        sym_instance.add("C")
        x = "A"
        prior = 0.5
        the = {'m': 0.000001}
        result = sym_instance.like(x, prior, the)
        expected_result = ((2) + 0.000001 * 0.5) / (4 + 0.000001)
        assert result == expected_result

    def test_sym_div(self):
        test_sym = SYM()
        for x in [1, 1, 1, 1, 2, 2, 3]:
            test_sym.add(x)
        e = test_sym.div()
        assert 0 < e < 158

    def test_bayes_diabetes(self):
        the = {'cohen': 0.35, 'file': 'data/diabetes.csv', 'help': 'False', 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'run_tc': 'all'}
        result = bayes(the)
        assert result > 70

    def test_bayes_weather(self):
        the = {'cohen': 0.35, 'file': 'data/weather.csv', 'help': 'False', 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'run_tc': 'all'}
        result = bayes(the)
        assert result > 70
    
    def test_bayes_soybean(self):
        the = {'cohen': 0.35, 'file': 'data/soybean.csv', 'help': 'False', 'k': 1.0, 'm': 2.0, 'seed': 31210.0, 'run_tc': 'all'}
        result = bayes(the)
        assert result > 70

    def test_shuffle_empty(self):
        test_list = []
        shuffled_list = shuffle(test_list)
        assert test_list == shuffled_list 

    def test_shuffle_num(self):
        test_list = [1, 2, 3, 4, 5]
        shuffled_list = shuffle(test_list)
        assert test_list != shuffled_list 

    def test_shuffle_sym(self):
        test_list = ['a', 'b', 'c', 'd', 'e']
        shuffled_list = shuffle(test_list)
        assert test_list != shuffled_list 

    def test_slice(self):
        sliced_list = slice([1, 2, 3, 4, 5])
        assert [2,3,4,5] ==  sliced_list

    def test_slice_start_stop(self):
        sliced_list = slice([1, 2, 3, 4, 5], 1, 4)
        assert sliced_list == [2, 3, 4] 

    def test_slice_neg_start_stop(self):
        sliced_list = slice([1, 2, 3, 4, 5], -3, -1)
        assert sliced_list == [3, 4]  
    
    def test_slice_start_stop_incr(self):
        sliced_list = slice( [1, 2, 3, 4, 5], 0, 5, 2)
        assert sliced_list == [2,4] 
    
    def run(self):
        print("-------------------Test Results--------------------")
        test_functions = [func for func in dir(self) if func.startswith('test_') and callable(getattr(self, func))]
        for test_func_name in test_functions:
            test_func = getattr(self, test_func_name)
            try:
                test_func()
                print("Test "+test_func_name+" passed")
            except Exception as e:
                print("Test "+test_func_name+" failed: "+str(e))

    def inp_test(self,inp,inp_test_map):
        if inp['test'] == 'all':
            self.run()
        elif inp['test'] != None and inp['test'] != '':
            print("-------------------Test Results--------------------")
            try:
                inp_test_map[inp['test']]()
                print('Test '+inp['test']+' passed')
            except AssertionError as e:
                print('Test '+inp['test']+' failed: '+str(e))
        else:
            pass

if __name__ == '__main__':
    # Configuring input arguments
    test_suite = Tests()
    inp_test_map = {name[5:]: getattr(test_suite, name) for name in dir(test_suite) if name.startswith('test_')}

    inp, s_inp = settings(help_str)
    test_suite = Tests()
    test_suite.inp_test(inp, inp_test_map)

    