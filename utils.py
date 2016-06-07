import math
def decimal_to_ternary(d, sz):
	# sz = math.ceil(math.pow(d, 1.0/3.0))
    t = [0] * sz
    for i in range(sz-1, -1, -1):
        r = d % 3
        t[i] = r 
        
        if d < 3:
        	return t
        
        d = math.floor(d/3)

    return t


def test_decimal_to_ternary():
	t = decimal_to_ternary(25, 3)
	assert(t == [2,2,1])
	t = decimal_to_ternary(1, 3)
	assert(t == [0,0,1])
	t = decimal_to_ternary(10, 3)
	assert(t == [1,0,1])
	print('decimal_to_ternary passed')

if __name__ == '__main__':
    test_decimal_to_ternary()