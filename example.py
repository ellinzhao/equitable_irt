from classes.subject import Subject
from classes.rgb_ir_data import RGB_IR_Data


# Do not need to construct an instance of RGB_IR_Data directly
data = RGB_IR_Data(data_path, 70, 40)

sub1 = Subject(0, csv_path, data_path)
