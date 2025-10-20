'''
정보 print해서 보고싶을때 간단히 사용하는 코드
'''
import inspect
from trl import SFTTrainer
print(inspect.signature(SFTTrainer.__init__))