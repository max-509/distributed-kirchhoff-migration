??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??
?
!travel_times_nn_3/dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*2
shared_name#!travel_times_nn_3/dense_35/kernel
?
5travel_times_nn_3/dense_35/kernel/Read/ReadVariableOpReadVariableOp!travel_times_nn_3/dense_35/kernel*
_output_shapes

:2*
dtype0
?
travel_times_nn_3/dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!travel_times_nn_3/dense_35/bias
?
3travel_times_nn_3/dense_35/bias/Read/ReadVariableOpReadVariableOptravel_times_nn_3/dense_35/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
/travel_times_nn_3/residualblock/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*@
shared_name1/travel_times_nn_3/residualblock/dense_17/kernel
?
Ctravel_times_nn_3/residualblock/dense_17/kernel/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock/dense_17/kernel*
_output_shapes

:2*
dtype0
?
-travel_times_nn_3/residualblock/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-travel_times_nn_3/residualblock/dense_17/bias
?
Atravel_times_nn_3/residualblock/dense_17/bias/Read/ReadVariableOpReadVariableOp-travel_times_nn_3/residualblock/dense_17/bias*
_output_shapes
:2*
dtype0
?
/travel_times_nn_3/residualblock/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*@
shared_name1/travel_times_nn_3/residualblock/dense_18/kernel
?
Ctravel_times_nn_3/residualblock/dense_18/kernel/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock/dense_18/kernel*
_output_shapes

:22*
dtype0
?
-travel_times_nn_3/residualblock/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-travel_times_nn_3/residualblock/dense_18/bias
?
Atravel_times_nn_3/residualblock/dense_18/bias/Read/ReadVariableOpReadVariableOp-travel_times_nn_3/residualblock/dense_18/bias*
_output_shapes
:2*
dtype0
?
/travel_times_nn_3/residualblock/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*@
shared_name1/travel_times_nn_3/residualblock/dense_19/kernel
?
Ctravel_times_nn_3/residualblock/dense_19/kernel/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock/dense_19/kernel*
_output_shapes

:2*
dtype0
?
-travel_times_nn_3/residualblock/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*>
shared_name/-travel_times_nn_3/residualblock/dense_19/bias
?
Atravel_times_nn_3/residualblock/dense_19/bias/Read/ReadVariableOpReadVariableOp-travel_times_nn_3/residualblock/dense_19/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_1/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_1/dense_20/kernel
?
Etravel_times_nn_3/residualblock_1/dense_20/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_1/dense_20/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_1/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_1/dense_20/bias
?
Ctravel_times_nn_3/residualblock_1/dense_20/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_1/dense_20/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_1/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_1/dense_21/kernel
?
Etravel_times_nn_3/residualblock_1/dense_21/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_1/dense_21/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_1/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_1/dense_21/bias
?
Ctravel_times_nn_3/residualblock_1/dense_21/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_1/dense_21/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_1/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_1/dense_22/kernel
?
Etravel_times_nn_3/residualblock_1/dense_22/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_1/dense_22/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_1/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_1/dense_22/bias
?
Ctravel_times_nn_3/residualblock_1/dense_22/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_1/dense_22/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_2/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_2/dense_23/kernel
?
Etravel_times_nn_3/residualblock_2/dense_23/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_2/dense_23/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_2/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_2/dense_23/bias
?
Ctravel_times_nn_3/residualblock_2/dense_23/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_2/dense_23/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_2/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_2/dense_24/kernel
?
Etravel_times_nn_3/residualblock_2/dense_24/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_2/dense_24/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_2/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_2/dense_24/bias
?
Ctravel_times_nn_3/residualblock_2/dense_24/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_2/dense_24/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_2/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_2/dense_25/kernel
?
Etravel_times_nn_3/residualblock_2/dense_25/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_2/dense_25/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_2/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_2/dense_25/bias
?
Ctravel_times_nn_3/residualblock_2/dense_25/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_2/dense_25/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_3/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_3/dense_26/kernel
?
Etravel_times_nn_3/residualblock_3/dense_26/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_3/dense_26/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_3/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_3/dense_26/bias
?
Ctravel_times_nn_3/residualblock_3/dense_26/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_3/dense_26/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_3/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_3/dense_27/kernel
?
Etravel_times_nn_3/residualblock_3/dense_27/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_3/dense_27/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_3/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_3/dense_27/bias
?
Ctravel_times_nn_3/residualblock_3/dense_27/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_3/dense_27/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_3/dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_3/dense_28/kernel
?
Etravel_times_nn_3/residualblock_3/dense_28/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_3/dense_28/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_3/dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_3/dense_28/bias
?
Ctravel_times_nn_3/residualblock_3/dense_28/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_3/dense_28/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_4/dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_4/dense_29/kernel
?
Etravel_times_nn_3/residualblock_4/dense_29/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_4/dense_29/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_4/dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_4/dense_29/bias
?
Ctravel_times_nn_3/residualblock_4/dense_29/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_4/dense_29/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_4/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_4/dense_30/kernel
?
Etravel_times_nn_3/residualblock_4/dense_30/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_4/dense_30/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_4/dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_4/dense_30/bias
?
Ctravel_times_nn_3/residualblock_4/dense_30/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_4/dense_30/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_4/dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_4/dense_31/kernel
?
Etravel_times_nn_3/residualblock_4/dense_31/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_4/dense_31/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_4/dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_4/dense_31/bias
?
Ctravel_times_nn_3/residualblock_4/dense_31/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_4/dense_31/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_5/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_5/dense_32/kernel
?
Etravel_times_nn_3/residualblock_5/dense_32/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_5/dense_32/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_5/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_5/dense_32/bias
?
Ctravel_times_nn_3/residualblock_5/dense_32/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_5/dense_32/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_5/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_5/dense_33/kernel
?
Etravel_times_nn_3/residualblock_5/dense_33/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_5/dense_33/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_5/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_5/dense_33/bias
?
Ctravel_times_nn_3/residualblock_5/dense_33/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_5/dense_33/bias*
_output_shapes
:2*
dtype0
?
1travel_times_nn_3/residualblock_5/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*B
shared_name31travel_times_nn_3/residualblock_5/dense_34/kernel
?
Etravel_times_nn_3/residualblock_5/dense_34/kernel/Read/ReadVariableOpReadVariableOp1travel_times_nn_3/residualblock_5/dense_34/kernel*
_output_shapes

:22*
dtype0
?
/travel_times_nn_3/residualblock_5/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*@
shared_name1/travel_times_nn_3/residualblock_5/dense_34/bias
?
Ctravel_times_nn_3/residualblock_5/dense_34/bias/Read/ReadVariableOpReadVariableOp/travel_times_nn_3/residualblock_5/dense_34/bias*
_output_shapes
:2*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
(Adam/travel_times_nn_3/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*9
shared_name*(Adam/travel_times_nn_3/dense_35/kernel/m
?
<Adam/travel_times_nn_3/dense_35/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/travel_times_nn_3/dense_35/kernel/m*
_output_shapes

:2*
dtype0
?
&Adam/travel_times_nn_3/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/travel_times_nn_3/dense_35/bias/m
?
:Adam/travel_times_nn_3/dense_35/bias/m/Read/ReadVariableOpReadVariableOp&Adam/travel_times_nn_3/dense_35/bias/m*
_output_shapes
:*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_17/kernel/m
?
JAdam/travel_times_nn_3/residualblock/dense_17/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_17/kernel/m*
_output_shapes

:2*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_17/bias/m
?
HAdam/travel_times_nn_3/residualblock/dense_17/bias/m/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_17/bias/m*
_output_shapes
:2*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_18/kernel/m
?
JAdam/travel_times_nn_3/residualblock/dense_18/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_18/kernel/m*
_output_shapes

:22*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_18/bias/m
?
HAdam/travel_times_nn_3/residualblock/dense_18/bias/m/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_18/bias/m*
_output_shapes
:2*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_19/kernel/m
?
JAdam/travel_times_nn_3/residualblock/dense_19/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_19/kernel/m*
_output_shapes

:2*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_19/bias/m
?
HAdam/travel_times_nn_3/residualblock/dense_19/bias/m/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_19/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/m
?
LAdam/travel_times_nn_3/residualblock_1/dense_20/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_20/bias/m
?
JAdam/travel_times_nn_3/residualblock_1/dense_20/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/m
?
LAdam/travel_times_nn_3/residualblock_1/dense_21/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_21/bias/m
?
JAdam/travel_times_nn_3/residualblock_1/dense_21/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/m
?
LAdam/travel_times_nn_3/residualblock_1/dense_22/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_22/bias/m
?
JAdam/travel_times_nn_3/residualblock_1/dense_22/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/m
?
LAdam/travel_times_nn_3/residualblock_2/dense_23/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_23/bias/m
?
JAdam/travel_times_nn_3/residualblock_2/dense_23/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/m
?
LAdam/travel_times_nn_3/residualblock_2/dense_24/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_24/bias/m
?
JAdam/travel_times_nn_3/residualblock_2/dense_24/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/m
?
LAdam/travel_times_nn_3/residualblock_2/dense_25/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_25/bias/m
?
JAdam/travel_times_nn_3/residualblock_2/dense_25/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/m
?
LAdam/travel_times_nn_3/residualblock_3/dense_26/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_26/bias/m
?
JAdam/travel_times_nn_3/residualblock_3/dense_26/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/m
?
LAdam/travel_times_nn_3/residualblock_3/dense_27/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_27/bias/m
?
JAdam/travel_times_nn_3/residualblock_3/dense_27/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/m
?
LAdam/travel_times_nn_3/residualblock_3/dense_28/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_28/bias/m
?
JAdam/travel_times_nn_3/residualblock_3/dense_28/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/m
?
LAdam/travel_times_nn_3/residualblock_4/dense_29/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_29/bias/m
?
JAdam/travel_times_nn_3/residualblock_4/dense_29/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/m
?
LAdam/travel_times_nn_3/residualblock_4/dense_30/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_30/bias/m
?
JAdam/travel_times_nn_3/residualblock_4/dense_30/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/m
?
LAdam/travel_times_nn_3/residualblock_4/dense_31/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_31/bias/m
?
JAdam/travel_times_nn_3/residualblock_4/dense_31/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/m
?
LAdam/travel_times_nn_3/residualblock_5/dense_32/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_32/bias/m
?
JAdam/travel_times_nn_3/residualblock_5/dense_32/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/m
?
LAdam/travel_times_nn_3/residualblock_5/dense_33/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_33/bias/m
?
JAdam/travel_times_nn_3/residualblock_5/dense_33/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/m*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/m
?
LAdam/travel_times_nn_3/residualblock_5/dense_34/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/m*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_34/bias/m
?
JAdam/travel_times_nn_3/residualblock_5/dense_34/bias/m/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/m*
_output_shapes
:2*
dtype0
?
(Adam/travel_times_nn_3/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*9
shared_name*(Adam/travel_times_nn_3/dense_35/kernel/v
?
<Adam/travel_times_nn_3/dense_35/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/travel_times_nn_3/dense_35/kernel/v*
_output_shapes

:2*
dtype0
?
&Adam/travel_times_nn_3/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/travel_times_nn_3/dense_35/bias/v
?
:Adam/travel_times_nn_3/dense_35/bias/v/Read/ReadVariableOpReadVariableOp&Adam/travel_times_nn_3/dense_35/bias/v*
_output_shapes
:*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_17/kernel/v
?
JAdam/travel_times_nn_3/residualblock/dense_17/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_17/kernel/v*
_output_shapes

:2*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_17/bias/v
?
HAdam/travel_times_nn_3/residualblock/dense_17/bias/v/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_17/bias/v*
_output_shapes
:2*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_18/kernel/v
?
JAdam/travel_times_nn_3/residualblock/dense_18/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_18/kernel/v*
_output_shapes

:22*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_18/bias/v
?
HAdam/travel_times_nn_3/residualblock/dense_18/bias/v/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_18/bias/v*
_output_shapes
:2*
dtype0
?
6Adam/travel_times_nn_3/residualblock/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*G
shared_name86Adam/travel_times_nn_3/residualblock/dense_19/kernel/v
?
JAdam/travel_times_nn_3/residualblock/dense_19/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock/dense_19/kernel/v*
_output_shapes

:2*
dtype0
?
4Adam/travel_times_nn_3/residualblock/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*E
shared_name64Adam/travel_times_nn_3/residualblock/dense_19/bias/v
?
HAdam/travel_times_nn_3/residualblock/dense_19/bias/v/Read/ReadVariableOpReadVariableOp4Adam/travel_times_nn_3/residualblock/dense_19/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/v
?
LAdam/travel_times_nn_3/residualblock_1/dense_20/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_20/bias/v
?
JAdam/travel_times_nn_3/residualblock_1/dense_20/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/v
?
LAdam/travel_times_nn_3/residualblock_1/dense_21/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_21/bias/v
?
JAdam/travel_times_nn_3/residualblock_1/dense_21/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/v
?
LAdam/travel_times_nn_3/residualblock_1/dense_22/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_1/dense_22/bias/v
?
JAdam/travel_times_nn_3/residualblock_1/dense_22/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/v
?
LAdam/travel_times_nn_3/residualblock_2/dense_23/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_23/bias/v
?
JAdam/travel_times_nn_3/residualblock_2/dense_23/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/v
?
LAdam/travel_times_nn_3/residualblock_2/dense_24/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_24/bias/v
?
JAdam/travel_times_nn_3/residualblock_2/dense_24/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/v
?
LAdam/travel_times_nn_3/residualblock_2/dense_25/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_2/dense_25/bias/v
?
JAdam/travel_times_nn_3/residualblock_2/dense_25/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/v
?
LAdam/travel_times_nn_3/residualblock_3/dense_26/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_26/bias/v
?
JAdam/travel_times_nn_3/residualblock_3/dense_26/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/v
?
LAdam/travel_times_nn_3/residualblock_3/dense_27/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_27/bias/v
?
JAdam/travel_times_nn_3/residualblock_3/dense_27/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/v
?
LAdam/travel_times_nn_3/residualblock_3/dense_28/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_3/dense_28/bias/v
?
JAdam/travel_times_nn_3/residualblock_3/dense_28/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/v
?
LAdam/travel_times_nn_3/residualblock_4/dense_29/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_29/bias/v
?
JAdam/travel_times_nn_3/residualblock_4/dense_29/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/v
?
LAdam/travel_times_nn_3/residualblock_4/dense_30/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_30/bias/v
?
JAdam/travel_times_nn_3/residualblock_4/dense_30/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/v
?
LAdam/travel_times_nn_3/residualblock_4/dense_31/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_4/dense_31/bias/v
?
JAdam/travel_times_nn_3/residualblock_4/dense_31/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/v
?
LAdam/travel_times_nn_3/residualblock_5/dense_32/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_32/bias/v
?
JAdam/travel_times_nn_3/residualblock_5/dense_32/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/v
?
LAdam/travel_times_nn_3/residualblock_5/dense_33/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_33/bias/v
?
JAdam/travel_times_nn_3/residualblock_5/dense_33/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/v*
_output_shapes
:2*
dtype0
?
8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*I
shared_name:8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/v
?
LAdam/travel_times_nn_3/residualblock_5/dense_34/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/v*
_output_shapes

:22*
dtype0
?
6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*G
shared_name86Adam/travel_times_nn_3/residualblock_5/dense_34/bias/v
?
JAdam/travel_times_nn_3/residualblock_5/dense_34/bias/v/Read/ReadVariableOpReadVariableOp6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
input_layer
	rescaling

denses
output_layer
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
*
0
1
2
3
4
5
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem?m? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
420
521
622
723
824
925
:26
;27
<28
=29
>30
?31
@32
A33
B34
C35
36
37
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
420
521
622
723
824
925
:26
;27
<28
=29
>30
?31
@32
A33
B34
C35
36
37
 
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
?
Ndense_1
Odense_2
Pdense_3
Qadd
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
?
Vdense_1
Wdense_2
Xdense_3
Yadd
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?
^dense_1
_dense_2
`dense_3
aadd
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?
fdense_1
gdense_2
hdense_3
iadd
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
?
ndense_1
odense_2
pdense_3
qadd
r	variables
strainable_variables
tregularization_losses
u	keras_api
?
vdense_1
wdense_2
xdense_3
yadd
z	variables
{trainable_variables
|regularization_losses
}	keras_api
ec
VARIABLE_VALUE!travel_times_nn_3/dense_35/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtravel_times_nn_3/dense_35/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/travel_times_nn_3/residualblock/dense_17/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-travel_times_nn_3/residualblock/dense_17/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/travel_times_nn_3/residualblock/dense_18/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-travel_times_nn_3/residualblock/dense_18/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/travel_times_nn_3/residualblock/dense_19/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE-travel_times_nn_3/residualblock/dense_19/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1travel_times_nn_3/residualblock_1/dense_20/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/travel_times_nn_3/residualblock_1/dense_20/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1travel_times_nn_3/residualblock_1/dense_21/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/travel_times_nn_3/residualblock_1/dense_21/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_1/dense_22/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_1/dense_22/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_2/dense_23/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_2/dense_23/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_2/dense_24/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_2/dense_24/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_2/dense_25/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_2/dense_25/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_3/dense_26/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_3/dense_26/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_3/dense_27/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_3/dense_27/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_3/dense_28/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_3/dense_28/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_4/dense_29/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_4/dense_29/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_4/dense_30/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_4/dense_30/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_4/dense_31/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_4/dense_31/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_5/dense_32/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_5/dense_32/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_5/dense_33/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_5/dense_33/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1travel_times_nn_3/residualblock_5/dense_34/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/travel_times_nn_3/residualblock_5/dense_34/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
8

?0
?1
?2
 
 
 
 
 
 
 
l

 kernel
!bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

"kernel
#bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

$kernel
%bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
 0
!1
"2
#3
$4
%5
*
 0
!1
"2
#3
$4
%5
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
l

&kernel
'bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

(kernel
)bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

*kernel
+bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
&0
'1
(2
)3
*4
+5
*
&0
'1
(2
)3
*4
+5
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
l

,kernel
-bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

.kernel
/bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

0kernel
1bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
,0
-1
.2
/3
04
15
*
,0
-1
.2
/3
04
15
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
l

2kernel
3bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

4kernel
5bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

6kernel
7bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
20
31
42
53
64
75
*
20
31
42
53
64
75
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
l

8kernel
9bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

:kernel
;bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

<kernel
=bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
80
91
:2
;3
<4
=5
*
80
91
:2
;3
<4
=5
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
r	variables
strainable_variables
tregularization_losses
l

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
*
>0
?1
@2
A3
B4
C5
*
>0
?1
@2
A3
B4
C5
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api

 0
!1

 0
!1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

"0
#1

"0
#1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

$0
%1

$0
%1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

N0
O1
P2
Q3
 
 
 

&0
'1

&0
'1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

(0
)1

(0
)1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

*0
+1

*0
+1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

V0
W1
X2
Y3
 
 
 

,0
-1

,0
-1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

.0
/1

.0
/1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

00
11

00
11
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

^0
_1
`2
a3
 
 
 

20
31

20
31
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

40
51

40
51
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

60
71

60
71
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

f0
g1
h2
i3
 
 
 

80
91

80
91
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

:0
;1

:0
;1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

<0
=1

<0
=1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

n0
o1
p2
q3
 
 
 

>0
?1

>0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

@0
A1

@0
A1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

B0
C1

B0
C1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

v0
w1
x2
y3
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUE(Adam/travel_times_nn_3/dense_35/kernel/mJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/travel_times_nn_3/dense_35/bias/mHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_17/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_17/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_18/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_18/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_19/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_19/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(Adam/travel_times_nn_3/dense_35/kernel/vJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/travel_times_nn_3/dense_35/bias/vHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_17/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_17/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_18/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_18/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock/dense_19/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/travel_times_nn_3/residualblock/dense_19/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1/travel_times_nn_3/residualblock/dense_17/kernel-travel_times_nn_3/residualblock/dense_17/bias/travel_times_nn_3/residualblock/dense_18/kernel-travel_times_nn_3/residualblock/dense_18/bias/travel_times_nn_3/residualblock/dense_19/kernel-travel_times_nn_3/residualblock/dense_19/bias1travel_times_nn_3/residualblock_1/dense_20/kernel/travel_times_nn_3/residualblock_1/dense_20/bias1travel_times_nn_3/residualblock_1/dense_21/kernel/travel_times_nn_3/residualblock_1/dense_21/bias1travel_times_nn_3/residualblock_1/dense_22/kernel/travel_times_nn_3/residualblock_1/dense_22/bias1travel_times_nn_3/residualblock_2/dense_23/kernel/travel_times_nn_3/residualblock_2/dense_23/bias1travel_times_nn_3/residualblock_2/dense_24/kernel/travel_times_nn_3/residualblock_2/dense_24/bias1travel_times_nn_3/residualblock_2/dense_25/kernel/travel_times_nn_3/residualblock_2/dense_25/bias1travel_times_nn_3/residualblock_3/dense_26/kernel/travel_times_nn_3/residualblock_3/dense_26/bias1travel_times_nn_3/residualblock_3/dense_27/kernel/travel_times_nn_3/residualblock_3/dense_27/bias1travel_times_nn_3/residualblock_3/dense_28/kernel/travel_times_nn_3/residualblock_3/dense_28/bias1travel_times_nn_3/residualblock_4/dense_29/kernel/travel_times_nn_3/residualblock_4/dense_29/bias1travel_times_nn_3/residualblock_4/dense_30/kernel/travel_times_nn_3/residualblock_4/dense_30/bias1travel_times_nn_3/residualblock_4/dense_31/kernel/travel_times_nn_3/residualblock_4/dense_31/bias1travel_times_nn_3/residualblock_5/dense_32/kernel/travel_times_nn_3/residualblock_5/dense_32/bias1travel_times_nn_3/residualblock_5/dense_33/kernel/travel_times_nn_3/residualblock_5/dense_33/bias1travel_times_nn_3/residualblock_5/dense_34/kernel/travel_times_nn_3/residualblock_5/dense_34/bias!travel_times_nn_3/dense_35/kerneltravel_times_nn_3/dense_35/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_3151837
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?G
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5travel_times_nn_3/dense_35/kernel/Read/ReadVariableOp3travel_times_nn_3/dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpCtravel_times_nn_3/residualblock/dense_17/kernel/Read/ReadVariableOpAtravel_times_nn_3/residualblock/dense_17/bias/Read/ReadVariableOpCtravel_times_nn_3/residualblock/dense_18/kernel/Read/ReadVariableOpAtravel_times_nn_3/residualblock/dense_18/bias/Read/ReadVariableOpCtravel_times_nn_3/residualblock/dense_19/kernel/Read/ReadVariableOpAtravel_times_nn_3/residualblock/dense_19/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_1/dense_20/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_1/dense_20/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_1/dense_21/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_1/dense_21/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_1/dense_22/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_1/dense_22/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_2/dense_23/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_2/dense_23/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_2/dense_24/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_2/dense_24/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_2/dense_25/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_2/dense_25/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_3/dense_26/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_3/dense_26/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_3/dense_27/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_3/dense_27/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_3/dense_28/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_3/dense_28/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_4/dense_29/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_4/dense_29/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_4/dense_30/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_4/dense_30/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_4/dense_31/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_4/dense_31/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_5/dense_32/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_5/dense_32/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_5/dense_33/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_5/dense_33/bias/Read/ReadVariableOpEtravel_times_nn_3/residualblock_5/dense_34/kernel/Read/ReadVariableOpCtravel_times_nn_3/residualblock_5/dense_34/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp<Adam/travel_times_nn_3/dense_35/kernel/m/Read/ReadVariableOp:Adam/travel_times_nn_3/dense_35/bias/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_17/kernel/m/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_17/bias/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_18/kernel/m/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_18/bias/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_19/kernel/m/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_19/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_20/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_20/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_21/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_21/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_22/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_22/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_23/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_23/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_24/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_24/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_25/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_25/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_26/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_26/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_27/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_27/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_28/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_28/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_29/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_29/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_30/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_30/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_31/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_31/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_32/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_32/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_33/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_33/bias/m/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_34/kernel/m/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_34/bias/m/Read/ReadVariableOp<Adam/travel_times_nn_3/dense_35/kernel/v/Read/ReadVariableOp:Adam/travel_times_nn_3/dense_35/bias/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_17/kernel/v/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_17/bias/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_18/kernel/v/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_18/bias/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock/dense_19/kernel/v/Read/ReadVariableOpHAdam/travel_times_nn_3/residualblock/dense_19/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_20/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_20/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_21/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_21/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_1/dense_22/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_1/dense_22/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_23/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_23/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_24/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_24/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_2/dense_25/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_2/dense_25/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_26/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_26/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_27/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_27/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_3/dense_28/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_3/dense_28/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_29/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_29/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_30/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_30/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_4/dense_31/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_4/dense_31/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_32/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_32/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_33/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_33/bias/v/Read/ReadVariableOpLAdam/travel_times_nn_3/residualblock_5/dense_34/kernel/v/Read/ReadVariableOpJAdam/travel_times_nn_3/residualblock_5/dense_34/bias/v/Read/ReadVariableOpConst*?
Tin?
?2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_3153007
?3
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!travel_times_nn_3/dense_35/kerneltravel_times_nn_3/dense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate/travel_times_nn_3/residualblock/dense_17/kernel-travel_times_nn_3/residualblock/dense_17/bias/travel_times_nn_3/residualblock/dense_18/kernel-travel_times_nn_3/residualblock/dense_18/bias/travel_times_nn_3/residualblock/dense_19/kernel-travel_times_nn_3/residualblock/dense_19/bias1travel_times_nn_3/residualblock_1/dense_20/kernel/travel_times_nn_3/residualblock_1/dense_20/bias1travel_times_nn_3/residualblock_1/dense_21/kernel/travel_times_nn_3/residualblock_1/dense_21/bias1travel_times_nn_3/residualblock_1/dense_22/kernel/travel_times_nn_3/residualblock_1/dense_22/bias1travel_times_nn_3/residualblock_2/dense_23/kernel/travel_times_nn_3/residualblock_2/dense_23/bias1travel_times_nn_3/residualblock_2/dense_24/kernel/travel_times_nn_3/residualblock_2/dense_24/bias1travel_times_nn_3/residualblock_2/dense_25/kernel/travel_times_nn_3/residualblock_2/dense_25/bias1travel_times_nn_3/residualblock_3/dense_26/kernel/travel_times_nn_3/residualblock_3/dense_26/bias1travel_times_nn_3/residualblock_3/dense_27/kernel/travel_times_nn_3/residualblock_3/dense_27/bias1travel_times_nn_3/residualblock_3/dense_28/kernel/travel_times_nn_3/residualblock_3/dense_28/bias1travel_times_nn_3/residualblock_4/dense_29/kernel/travel_times_nn_3/residualblock_4/dense_29/bias1travel_times_nn_3/residualblock_4/dense_30/kernel/travel_times_nn_3/residualblock_4/dense_30/bias1travel_times_nn_3/residualblock_4/dense_31/kernel/travel_times_nn_3/residualblock_4/dense_31/bias1travel_times_nn_3/residualblock_5/dense_32/kernel/travel_times_nn_3/residualblock_5/dense_32/bias1travel_times_nn_3/residualblock_5/dense_33/kernel/travel_times_nn_3/residualblock_5/dense_33/bias1travel_times_nn_3/residualblock_5/dense_34/kernel/travel_times_nn_3/residualblock_5/dense_34/biastotalcounttotal_1count_1total_2count_2(Adam/travel_times_nn_3/dense_35/kernel/m&Adam/travel_times_nn_3/dense_35/bias/m6Adam/travel_times_nn_3/residualblock/dense_17/kernel/m4Adam/travel_times_nn_3/residualblock/dense_17/bias/m6Adam/travel_times_nn_3/residualblock/dense_18/kernel/m4Adam/travel_times_nn_3/residualblock/dense_18/bias/m6Adam/travel_times_nn_3/residualblock/dense_19/kernel/m4Adam/travel_times_nn_3/residualblock/dense_19/bias/m8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/m6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/m8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/m6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/m8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/m6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/m8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/m6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/m8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/m6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/m8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/m6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/m8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/m6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/m8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/m6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/m8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/m6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/m8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/m6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/m8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/m6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/m8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/m6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/m8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/m6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/m8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/m6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/m8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/m6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/m(Adam/travel_times_nn_3/dense_35/kernel/v&Adam/travel_times_nn_3/dense_35/bias/v6Adam/travel_times_nn_3/residualblock/dense_17/kernel/v4Adam/travel_times_nn_3/residualblock/dense_17/bias/v6Adam/travel_times_nn_3/residualblock/dense_18/kernel/v4Adam/travel_times_nn_3/residualblock/dense_18/bias/v6Adam/travel_times_nn_3/residualblock/dense_19/kernel/v4Adam/travel_times_nn_3/residualblock/dense_19/bias/v8Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/v6Adam/travel_times_nn_3/residualblock_1/dense_20/bias/v8Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/v6Adam/travel_times_nn_3/residualblock_1/dense_21/bias/v8Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/v6Adam/travel_times_nn_3/residualblock_1/dense_22/bias/v8Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/v6Adam/travel_times_nn_3/residualblock_2/dense_23/bias/v8Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/v6Adam/travel_times_nn_3/residualblock_2/dense_24/bias/v8Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/v6Adam/travel_times_nn_3/residualblock_2/dense_25/bias/v8Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/v6Adam/travel_times_nn_3/residualblock_3/dense_26/bias/v8Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/v6Adam/travel_times_nn_3/residualblock_3/dense_27/bias/v8Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/v6Adam/travel_times_nn_3/residualblock_3/dense_28/bias/v8Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/v6Adam/travel_times_nn_3/residualblock_4/dense_29/bias/v8Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/v6Adam/travel_times_nn_3/residualblock_4/dense_30/bias/v8Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/v6Adam/travel_times_nn_3/residualblock_4/dense_31/bias/v8Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/v6Adam/travel_times_nn_3/residualblock_5/dense_32/bias/v8Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/v6Adam/travel_times_nn_3/residualblock_5/dense_33/bias/v8Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/v6Adam/travel_times_nn_3/residualblock_5/dense_34/bias/v*?
Tin?
?2~*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_3153392??
?
?
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3152567

inputs9
'dense_29_matmul_readvariableop_resource:226
(dense_29_biasadd_readvariableop_resource:29
'dense_30_matmul_readvariableop_resource:226
(dense_30_biasadd_readvariableop_resource:29
'dense_31_matmul_readvariableop_resource:226
(dense_31_biasadd_readvariableop_resource:2
identity??dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_29/MatMulMatMulinputs&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_29/TanhTanhdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_30/MatMulMatMuldense_29/Tanh:y:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_4/addAddV2dense_30/BiasAdd:output:0dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_4/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
ة
?/
"__inference__wrapped_model_3150688
input_1Y
Gtravel_times_nn_3_residualblock_dense_17_matmul_readvariableop_resource:2V
Htravel_times_nn_3_residualblock_dense_17_biasadd_readvariableop_resource:2Y
Gtravel_times_nn_3_residualblock_dense_18_matmul_readvariableop_resource:22V
Htravel_times_nn_3_residualblock_dense_18_biasadd_readvariableop_resource:2Y
Gtravel_times_nn_3_residualblock_dense_19_matmul_readvariableop_resource:2V
Htravel_times_nn_3_residualblock_dense_19_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_1_dense_20_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_1_dense_20_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_1_dense_21_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_1_dense_21_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_1_dense_22_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_1_dense_22_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_2_dense_23_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_2_dense_23_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_2_dense_24_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_2_dense_24_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_2_dense_25_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_2_dense_25_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_3_dense_26_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_3_dense_26_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_3_dense_27_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_3_dense_27_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_3_dense_28_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_3_dense_28_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_4_dense_29_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_4_dense_29_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_4_dense_30_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_4_dense_30_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_4_dense_31_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_4_dense_31_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_5_dense_32_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_5_dense_32_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_5_dense_33_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_5_dense_33_biasadd_readvariableop_resource:2[
Itravel_times_nn_3_residualblock_5_dense_34_matmul_readvariableop_resource:22X
Jtravel_times_nn_3_residualblock_5_dense_34_biasadd_readvariableop_resource:2K
9travel_times_nn_3_dense_35_matmul_readvariableop_resource:2H
:travel_times_nn_3_dense_35_biasadd_readvariableop_resource:
identity??1travel_times_nn_3/dense_35/BiasAdd/ReadVariableOp?0travel_times_nn_3/dense_35/MatMul/ReadVariableOp??travel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOp?>travel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOp??travel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOp?>travel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOp??travel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOp?>travel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOp?Atravel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOp?@travel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOpm
$travel_times_nn_3/rescaling_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2????Mb0??
"travel_times_nn_3/rescaling_3/CastCast-travel_times_nn_3/rescaling_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: k
&travel_times_nn_3/rescaling_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!travel_times_nn_3/rescaling_3/mulMulinput_1&travel_times_nn_3/rescaling_3/Cast:y:0*
T0*'
_output_shapes
:??????????
!travel_times_nn_3/rescaling_3/addAddV2%travel_times_nn_3/rescaling_3/mul:z:0/travel_times_nn_3/rescaling_3/Cast_1/x:output:0*
T0*'
_output_shapes
:??????????
>travel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOpReadVariableOpGtravel_times_nn_3_residualblock_dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
/travel_times_nn_3/residualblock/dense_17/MatMulMatMul%travel_times_nn_3/rescaling_3/add:z:0Ftravel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
?travel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOpReadVariableOpHtravel_times_nn_3_residualblock_dense_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
0travel_times_nn_3/residualblock/dense_17/BiasAddBiasAdd9travel_times_nn_3/residualblock/dense_17/MatMul:product:0Gtravel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-travel_times_nn_3/residualblock/dense_17/TanhTanh9travel_times_nn_3/residualblock/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
>travel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOpReadVariableOpGtravel_times_nn_3_residualblock_dense_18_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
/travel_times_nn_3/residualblock/dense_18/MatMulMatMul1travel_times_nn_3/residualblock/dense_17/Tanh:y:0Ftravel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
?travel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOpReadVariableOpHtravel_times_nn_3_residualblock_dense_18_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
0travel_times_nn_3/residualblock/dense_18/BiasAddBiasAdd9travel_times_nn_3/residualblock/dense_18/MatMul:product:0Gtravel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
>travel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOpReadVariableOpGtravel_times_nn_3_residualblock_dense_19_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
/travel_times_nn_3/residualblock/dense_19/MatMulMatMul%travel_times_nn_3/rescaling_3/add:z:0Ftravel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
?travel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOpReadVariableOpHtravel_times_nn_3_residualblock_dense_19_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
0travel_times_nn_3/residualblock/dense_19/BiasAddBiasAdd9travel_times_nn_3/residualblock/dense_19/MatMul:product:0Gtravel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
'travel_times_nn_3/residualblock/add/addAddV29travel_times_nn_3/residualblock/dense_18/BiasAdd:output:09travel_times_nn_3/residualblock/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
$travel_times_nn_3/residualblock/TanhTanh+travel_times_nn_3/residualblock/add/add:z:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_1_dense_20_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_1/dense_20/MatMulMatMul(travel_times_nn_3/residualblock/Tanh:y:0Htravel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_1_dense_20_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_1/dense_20/BiasAddBiasAdd;travel_times_nn_3/residualblock_1/dense_20/MatMul:product:0Itravel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/travel_times_nn_3/residualblock_1/dense_20/TanhTanh;travel_times_nn_3/residualblock_1/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_1_dense_21_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_1/dense_21/MatMulMatMul3travel_times_nn_3/residualblock_1/dense_20/Tanh:y:0Htravel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_1_dense_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_1/dense_21/BiasAddBiasAdd;travel_times_nn_3/residualblock_1/dense_21/MatMul:product:0Itravel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_1_dense_22_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_1/dense_22/MatMulMatMul(travel_times_nn_3/residualblock/Tanh:y:0Htravel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_1_dense_22_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_1/dense_22/BiasAddBiasAdd;travel_times_nn_3/residualblock_1/dense_22/MatMul:product:0Itravel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
+travel_times_nn_3/residualblock_1/add_1/addAddV2;travel_times_nn_3/residualblock_1/dense_21/BiasAdd:output:0;travel_times_nn_3/residualblock_1/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
&travel_times_nn_3/residualblock_1/TanhTanh/travel_times_nn_3/residualblock_1/add_1/add:z:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_2_dense_23_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_2/dense_23/MatMulMatMul*travel_times_nn_3/residualblock_1/Tanh:y:0Htravel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_2/dense_23/BiasAddBiasAdd;travel_times_nn_3/residualblock_2/dense_23/MatMul:product:0Itravel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/travel_times_nn_3/residualblock_2/dense_23/TanhTanh;travel_times_nn_3/residualblock_2/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_2_dense_24_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_2/dense_24/MatMulMatMul3travel_times_nn_3/residualblock_2/dense_23/Tanh:y:0Htravel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_2_dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_2/dense_24/BiasAddBiasAdd;travel_times_nn_3/residualblock_2/dense_24/MatMul:product:0Itravel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_2_dense_25_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_2/dense_25/MatMulMatMul*travel_times_nn_3/residualblock_1/Tanh:y:0Htravel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_2_dense_25_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_2/dense_25/BiasAddBiasAdd;travel_times_nn_3/residualblock_2/dense_25/MatMul:product:0Itravel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
+travel_times_nn_3/residualblock_2/add_2/addAddV2;travel_times_nn_3/residualblock_2/dense_24/BiasAdd:output:0;travel_times_nn_3/residualblock_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
&travel_times_nn_3/residualblock_2/TanhTanh/travel_times_nn_3/residualblock_2/add_2/add:z:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_3/dense_26/MatMulMatMul*travel_times_nn_3/residualblock_2/Tanh:y:0Htravel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_3/dense_26/BiasAddBiasAdd;travel_times_nn_3/residualblock_3/dense_26/MatMul:product:0Itravel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/travel_times_nn_3/residualblock_3/dense_26/TanhTanh;travel_times_nn_3/residualblock_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_3/dense_27/MatMulMatMul3travel_times_nn_3/residualblock_3/dense_26/Tanh:y:0Htravel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_3/dense_27/BiasAddBiasAdd;travel_times_nn_3/residualblock_3/dense_27/MatMul:product:0Itravel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_3_dense_28_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_3/dense_28/MatMulMatMul*travel_times_nn_3/residualblock_2/Tanh:y:0Htravel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_3_dense_28_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_3/dense_28/BiasAddBiasAdd;travel_times_nn_3/residualblock_3/dense_28/MatMul:product:0Itravel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
+travel_times_nn_3/residualblock_3/add_3/addAddV2;travel_times_nn_3/residualblock_3/dense_27/BiasAdd:output:0;travel_times_nn_3/residualblock_3/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
&travel_times_nn_3/residualblock_3/TanhTanh/travel_times_nn_3/residualblock_3/add_3/add:z:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_4_dense_29_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_4/dense_29/MatMulMatMul*travel_times_nn_3/residualblock_3/Tanh:y:0Htravel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_4_dense_29_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_4/dense_29/BiasAddBiasAdd;travel_times_nn_3/residualblock_4/dense_29/MatMul:product:0Itravel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/travel_times_nn_3/residualblock_4/dense_29/TanhTanh;travel_times_nn_3/residualblock_4/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_4_dense_30_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_4/dense_30/MatMulMatMul3travel_times_nn_3/residualblock_4/dense_29/Tanh:y:0Htravel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_4_dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_4/dense_30/BiasAddBiasAdd;travel_times_nn_3/residualblock_4/dense_30/MatMul:product:0Itravel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_4_dense_31_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_4/dense_31/MatMulMatMul*travel_times_nn_3/residualblock_3/Tanh:y:0Htravel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_4_dense_31_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_4/dense_31/BiasAddBiasAdd;travel_times_nn_3/residualblock_4/dense_31/MatMul:product:0Itravel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
+travel_times_nn_3/residualblock_4/add_4/addAddV2;travel_times_nn_3/residualblock_4/dense_30/BiasAdd:output:0;travel_times_nn_3/residualblock_4/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
&travel_times_nn_3/residualblock_4/TanhTanh/travel_times_nn_3/residualblock_4/add_4/add:z:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_5_dense_32_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_5/dense_32/MatMulMatMul*travel_times_nn_3/residualblock_4/Tanh:y:0Htravel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_5_dense_32_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_5/dense_32/BiasAddBiasAdd;travel_times_nn_3/residualblock_5/dense_32/MatMul:product:0Itravel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/travel_times_nn_3/residualblock_5/dense_32/TanhTanh;travel_times_nn_3/residualblock_5/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_5_dense_33_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_5/dense_33/MatMulMatMul3travel_times_nn_3/residualblock_5/dense_32/Tanh:y:0Htravel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_5_dense_33_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_5/dense_33/BiasAddBiasAdd;travel_times_nn_3/residualblock_5/dense_33/MatMul:product:0Itravel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
@travel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOpReadVariableOpItravel_times_nn_3_residualblock_5_dense_34_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
1travel_times_nn_3/residualblock_5/dense_34/MatMulMatMul*travel_times_nn_3/residualblock_4/Tanh:y:0Htravel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
Atravel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOpReadVariableOpJtravel_times_nn_3_residualblock_5_dense_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
2travel_times_nn_3/residualblock_5/dense_34/BiasAddBiasAdd;travel_times_nn_3/residualblock_5/dense_34/MatMul:product:0Itravel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
+travel_times_nn_3/residualblock_5/add_5/addAddV2;travel_times_nn_3/residualblock_5/dense_33/BiasAdd:output:0;travel_times_nn_3/residualblock_5/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
&travel_times_nn_3/residualblock_5/TanhTanh/travel_times_nn_3/residualblock_5/add_5/add:z:0*
T0*'
_output_shapes
:?????????2?
0travel_times_nn_3/dense_35/MatMul/ReadVariableOpReadVariableOp9travel_times_nn_3_dense_35_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
!travel_times_nn_3/dense_35/MatMulMatMul*travel_times_nn_3/residualblock_5/Tanh:y:08travel_times_nn_3/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1travel_times_nn_3/dense_35/BiasAdd/ReadVariableOpReadVariableOp:travel_times_nn_3_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"travel_times_nn_3/dense_35/BiasAddBiasAdd+travel_times_nn_3/dense_35/MatMul:product:09travel_times_nn_3/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
travel_times_nn_3/dense_35/ExpExp+travel_times_nn_3/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????v
%travel_times_nn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'travel_times_nn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'travel_times_nn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
travel_times_nn_3/strided_sliceStridedSlice%travel_times_nn_3/rescaling_3/add:z:0.travel_times_nn_3/strided_slice/stack:output:00travel_times_nn_3/strided_slice/stack_1:output:00travel_times_nn_3/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskx
'travel_times_nn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)travel_times_nn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)travel_times_nn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!travel_times_nn_3/strided_slice_1StridedSlice%travel_times_nn_3/rescaling_3/add:z:00travel_times_nn_3/strided_slice_1/stack:output:02travel_times_nn_3/strided_slice_1/stack_1:output:02travel_times_nn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
travel_times_nn_3/SubSub(travel_times_nn_3/strided_slice:output:0*travel_times_nn_3/strided_slice_1:output:0*
T0*#
_output_shapes
:?????????k
travel_times_nn_3/SquareSquaretravel_times_nn_3/Sub:z:0*
T0*#
_output_shapes
:?????????x
'travel_times_nn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)travel_times_nn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)travel_times_nn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!travel_times_nn_3/strided_slice_2StridedSlice%travel_times_nn_3/rescaling_3/add:z:00travel_times_nn_3/strided_slice_2/stack:output:02travel_times_nn_3/strided_slice_2/stack_1:output:02travel_times_nn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
travel_times_nn_3/Square_1Square*travel_times_nn_3/strided_slice_2:output:0*
T0*#
_output_shapes
:??????????
travel_times_nn_3/AddAddV2travel_times_nn_3/Square:y:0travel_times_nn_3/Square_1:y:0*
T0*#
_output_shapes
:?????????g
travel_times_nn_3/SqrtSqrttravel_times_nn_3/Add:z:0*
T0*#
_output_shapes
:?????????p
travel_times_nn_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
travel_times_nn_3/ReshapeReshapetravel_times_nn_3/Sqrt:y:0(travel_times_nn_3/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
travel_times_nn_3/MulMul"travel_times_nn_3/dense_35/Exp:y:0"travel_times_nn_3/Reshape:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitytravel_times_nn_3/Mul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp2^travel_times_nn_3/dense_35/BiasAdd/ReadVariableOp1^travel_times_nn_3/dense_35/MatMul/ReadVariableOp@^travel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOp?^travel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOp@^travel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOp?^travel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOp@^travel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOp?^travel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOpB^travel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOpA^travel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1travel_times_nn_3/dense_35/BiasAdd/ReadVariableOp1travel_times_nn_3/dense_35/BiasAdd/ReadVariableOp2d
0travel_times_nn_3/dense_35/MatMul/ReadVariableOp0travel_times_nn_3/dense_35/MatMul/ReadVariableOp2?
?travel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOp?travel_times_nn_3/residualblock/dense_17/BiasAdd/ReadVariableOp2?
>travel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOp>travel_times_nn_3/residualblock/dense_17/MatMul/ReadVariableOp2?
?travel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOp?travel_times_nn_3/residualblock/dense_18/BiasAdd/ReadVariableOp2?
>travel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOp>travel_times_nn_3/residualblock/dense_18/MatMul/ReadVariableOp2?
?travel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOp?travel_times_nn_3/residualblock/dense_19/BiasAdd/ReadVariableOp2?
>travel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOp>travel_times_nn_3/residualblock/dense_19/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_1/dense_20/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_1/dense_20/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_1/dense_21/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_1/dense_21/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_1/dense_22/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_1/dense_22/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_2/dense_23/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_2/dense_23/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_2/dense_24/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_2/dense_24/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_2/dense_25/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_2/dense_25/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_3/dense_26/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_3/dense_26/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_3/dense_27/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_3/dense_27/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_3/dense_28/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_3/dense_28/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_4/dense_29/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_4/dense_29/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_4/dense_30/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_4/dense_30/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_4/dense_31/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_4/dense_31/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_5/dense_32/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_5/dense_32/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_5/dense_33/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_5/dense_33/MatMul/ReadVariableOp2?
Atravel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOpAtravel_times_nn_3/residualblock_5/dense_34/BiasAdd/ReadVariableOp2?
@travel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOp@travel_times_nn_3/residualblock_5/dense_34/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731

inputs9
'dense_17_matmul_readvariableop_resource:26
(dense_17_biasadd_readvariableop_resource:29
'dense_18_matmul_readvariableop_resource:226
(dense_18_biasadd_readvariableop_resource:29
'dense_19_matmul_readvariableop_resource:26
(dense_19_biasadd_readvariableop_resource:2
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0{
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_18/MatMulMatMuldense_17/Tanh:y:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0{
dense_19/MatMulMatMulinputs&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2x
add/addAddV2dense_18/BiasAdd:output:0dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2K
TanhTanhadd/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?	
3__inference_travel_times_nn_3_layer_call_fn_3151918

inputs
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:22
	unknown_6:2
	unknown_7:22
	unknown_8:2
	unknown_9:22

unknown_10:2

unknown_11:22

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:22

unknown_16:2

unknown_17:22

unknown_18:2

unknown_19:22

unknown_20:2

unknown_21:22

unknown_22:2

unknown_23:22

unknown_24:2

unknown_25:22

unknown_26:2

unknown_27:22

unknown_28:2

unknown_29:22

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:22

unknown_34:2

unknown_35:2

unknown_36:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3150978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848

inputs9
'dense_26_matmul_readvariableop_resource:226
(dense_26_biasadd_readvariableop_resource:29
'dense_27_matmul_readvariableop_resource:226
(dense_27_biasadd_readvariableop_resource:29
'dense_28_matmul_readvariableop_resource:226
(dense_28_biasadd_readvariableop_resource:2
identity??dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_26/MatMulMatMulinputs&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_26/TanhTanhdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_27/MatMulMatMuldense_26/Tanh:y:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_28/MatMulMatMulinputs&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_3/addAddV2dense_27/BiasAdd:output:0dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_3/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?$
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152161

inputsG
5residualblock_dense_17_matmul_readvariableop_resource:2D
6residualblock_dense_17_biasadd_readvariableop_resource:2G
5residualblock_dense_18_matmul_readvariableop_resource:22D
6residualblock_dense_18_biasadd_readvariableop_resource:2G
5residualblock_dense_19_matmul_readvariableop_resource:2D
6residualblock_dense_19_biasadd_readvariableop_resource:2I
7residualblock_1_dense_20_matmul_readvariableop_resource:22F
8residualblock_1_dense_20_biasadd_readvariableop_resource:2I
7residualblock_1_dense_21_matmul_readvariableop_resource:22F
8residualblock_1_dense_21_biasadd_readvariableop_resource:2I
7residualblock_1_dense_22_matmul_readvariableop_resource:22F
8residualblock_1_dense_22_biasadd_readvariableop_resource:2I
7residualblock_2_dense_23_matmul_readvariableop_resource:22F
8residualblock_2_dense_23_biasadd_readvariableop_resource:2I
7residualblock_2_dense_24_matmul_readvariableop_resource:22F
8residualblock_2_dense_24_biasadd_readvariableop_resource:2I
7residualblock_2_dense_25_matmul_readvariableop_resource:22F
8residualblock_2_dense_25_biasadd_readvariableop_resource:2I
7residualblock_3_dense_26_matmul_readvariableop_resource:22F
8residualblock_3_dense_26_biasadd_readvariableop_resource:2I
7residualblock_3_dense_27_matmul_readvariableop_resource:22F
8residualblock_3_dense_27_biasadd_readvariableop_resource:2I
7residualblock_3_dense_28_matmul_readvariableop_resource:22F
8residualblock_3_dense_28_biasadd_readvariableop_resource:2I
7residualblock_4_dense_29_matmul_readvariableop_resource:22F
8residualblock_4_dense_29_biasadd_readvariableop_resource:2I
7residualblock_4_dense_30_matmul_readvariableop_resource:22F
8residualblock_4_dense_30_biasadd_readvariableop_resource:2I
7residualblock_4_dense_31_matmul_readvariableop_resource:22F
8residualblock_4_dense_31_biasadd_readvariableop_resource:2I
7residualblock_5_dense_32_matmul_readvariableop_resource:22F
8residualblock_5_dense_32_biasadd_readvariableop_resource:2I
7residualblock_5_dense_33_matmul_readvariableop_resource:22F
8residualblock_5_dense_33_biasadd_readvariableop_resource:2I
7residualblock_5_dense_34_matmul_readvariableop_resource:22F
8residualblock_5_dense_34_biasadd_readvariableop_resource:29
'dense_35_matmul_readvariableop_resource:26
(dense_35_biasadd_readvariableop_resource:
identity??dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?-residualblock/dense_17/BiasAdd/ReadVariableOp?,residualblock/dense_17/MatMul/ReadVariableOp?-residualblock/dense_18/BiasAdd/ReadVariableOp?,residualblock/dense_18/MatMul/ReadVariableOp?-residualblock/dense_19/BiasAdd/ReadVariableOp?,residualblock/dense_19/MatMul/ReadVariableOp?/residualblock_1/dense_20/BiasAdd/ReadVariableOp?.residualblock_1/dense_20/MatMul/ReadVariableOp?/residualblock_1/dense_21/BiasAdd/ReadVariableOp?.residualblock_1/dense_21/MatMul/ReadVariableOp?/residualblock_1/dense_22/BiasAdd/ReadVariableOp?.residualblock_1/dense_22/MatMul/ReadVariableOp?/residualblock_2/dense_23/BiasAdd/ReadVariableOp?.residualblock_2/dense_23/MatMul/ReadVariableOp?/residualblock_2/dense_24/BiasAdd/ReadVariableOp?.residualblock_2/dense_24/MatMul/ReadVariableOp?/residualblock_2/dense_25/BiasAdd/ReadVariableOp?.residualblock_2/dense_25/MatMul/ReadVariableOp?/residualblock_3/dense_26/BiasAdd/ReadVariableOp?.residualblock_3/dense_26/MatMul/ReadVariableOp?/residualblock_3/dense_27/BiasAdd/ReadVariableOp?.residualblock_3/dense_27/MatMul/ReadVariableOp?/residualblock_3/dense_28/BiasAdd/ReadVariableOp?.residualblock_3/dense_28/MatMul/ReadVariableOp?/residualblock_4/dense_29/BiasAdd/ReadVariableOp?.residualblock_4/dense_29/MatMul/ReadVariableOp?/residualblock_4/dense_30/BiasAdd/ReadVariableOp?.residualblock_4/dense_30/MatMul/ReadVariableOp?/residualblock_4/dense_31/BiasAdd/ReadVariableOp?.residualblock_4/dense_31/MatMul/ReadVariableOp?/residualblock_5/dense_32/BiasAdd/ReadVariableOp?.residualblock_5/dense_32/MatMul/ReadVariableOp?/residualblock_5/dense_33/BiasAdd/ReadVariableOp?.residualblock_5/dense_33/MatMul/ReadVariableOp?/residualblock_5/dense_34/BiasAdd/ReadVariableOp?.residualblock_5/dense_34/MatMul/ReadVariableOp[
rescaling_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2????Mb0?e
rescaling_3/CastCastrescaling_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
rescaling_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    f
rescaling_3/mulMulinputsrescaling_3/Cast:y:0*
T0*'
_output_shapes
:?????????~
rescaling_3/addAddV2rescaling_3/mul:z:0rescaling_3/Cast_1/x:output:0*
T0*'
_output_shapes
:??????????
,residualblock/dense_17/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
residualblock/dense_17/MatMulMatMulrescaling_3/add:z:04residualblock/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_17/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_17/BiasAddBiasAdd'residualblock/dense_17/MatMul:product:05residualblock/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2~
residualblock/dense_17/TanhTanh'residualblock/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
,residualblock/dense_18/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_18_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock/dense_18/MatMulMatMulresidualblock/dense_17/Tanh:y:04residualblock/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_18/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_18_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_18/BiasAddBiasAdd'residualblock/dense_18/MatMul:product:05residualblock/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
,residualblock/dense_19/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_19_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
residualblock/dense_19/MatMulMatMulrescaling_3/add:z:04residualblock/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_19/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_19_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_19/BiasAddBiasAdd'residualblock/dense_19/MatMul:product:05residualblock/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock/add/addAddV2'residualblock/dense_18/BiasAdd:output:0'residualblock/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2g
residualblock/TanhTanhresidualblock/add/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_20/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_20_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_20/MatMulMatMulresidualblock/Tanh:y:06residualblock_1/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_20/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_20_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_20/BiasAddBiasAdd)residualblock_1/dense_20/MatMul:product:07residualblock_1/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_1/dense_20/TanhTanh)residualblock_1/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_21/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_21_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_21/MatMulMatMul!residualblock_1/dense_20/Tanh:y:06residualblock_1/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_21/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_21/BiasAddBiasAdd)residualblock_1/dense_21/MatMul:product:07residualblock_1/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_22/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_22_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_22/MatMulMatMulresidualblock/Tanh:y:06residualblock_1/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_22/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_22_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_22/BiasAddBiasAdd)residualblock_1/dense_22/MatMul:product:07residualblock_1/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_1/add_1/addAddV2)residualblock_1/dense_21/BiasAdd:output:0)residualblock_1/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_1/TanhTanhresidualblock_1/add_1/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_23/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_23_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_23/MatMulMatMulresidualblock_1/Tanh:y:06residualblock_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_23/BiasAddBiasAdd)residualblock_2/dense_23/MatMul:product:07residualblock_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_2/dense_23/TanhTanh)residualblock_2/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_24/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_24_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_24/MatMulMatMul!residualblock_2/dense_23/Tanh:y:06residualblock_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_24/BiasAddBiasAdd)residualblock_2/dense_24/MatMul:product:07residualblock_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_25/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_25_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_25/MatMulMatMulresidualblock_1/Tanh:y:06residualblock_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_25_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_25/BiasAddBiasAdd)residualblock_2/dense_25/MatMul:product:07residualblock_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_2/add_2/addAddV2)residualblock_2/dense_24/BiasAdd:output:0)residualblock_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_2/TanhTanhresidualblock_2/add_2/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_26/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_26/MatMulMatMulresidualblock_2/Tanh:y:06residualblock_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_26/BiasAddBiasAdd)residualblock_3/dense_26/MatMul:product:07residualblock_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_3/dense_26/TanhTanh)residualblock_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_27/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_27/MatMulMatMul!residualblock_3/dense_26/Tanh:y:06residualblock_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_27/BiasAddBiasAdd)residualblock_3/dense_27/MatMul:product:07residualblock_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_28/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_28_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_28/MatMulMatMulresidualblock_2/Tanh:y:06residualblock_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_28/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_28_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_28/BiasAddBiasAdd)residualblock_3/dense_28/MatMul:product:07residualblock_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_3/add_3/addAddV2)residualblock_3/dense_27/BiasAdd:output:0)residualblock_3/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_3/TanhTanhresidualblock_3/add_3/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_29/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_29_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_29/MatMulMatMulresidualblock_3/Tanh:y:06residualblock_4/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_29/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_29_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_29/BiasAddBiasAdd)residualblock_4/dense_29/MatMul:product:07residualblock_4/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_4/dense_29/TanhTanh)residualblock_4/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_30/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_30_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_30/MatMulMatMul!residualblock_4/dense_29/Tanh:y:06residualblock_4/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_30/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_30/BiasAddBiasAdd)residualblock_4/dense_30/MatMul:product:07residualblock_4/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_31/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_31_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_31/MatMulMatMulresidualblock_3/Tanh:y:06residualblock_4/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_31/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_31_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_31/BiasAddBiasAdd)residualblock_4/dense_31/MatMul:product:07residualblock_4/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_4/add_4/addAddV2)residualblock_4/dense_30/BiasAdd:output:0)residualblock_4/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_4/TanhTanhresidualblock_4/add_4/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_32/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_32_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_32/MatMulMatMulresidualblock_4/Tanh:y:06residualblock_5/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_32/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_32_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_32/BiasAddBiasAdd)residualblock_5/dense_32/MatMul:product:07residualblock_5/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_5/dense_32/TanhTanh)residualblock_5/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_33/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_33_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_33/MatMulMatMul!residualblock_5/dense_32/Tanh:y:06residualblock_5/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_33/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_33_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_33/BiasAddBiasAdd)residualblock_5/dense_33/MatMul:product:07residualblock_5/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_34/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_34_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_34/MatMulMatMulresidualblock_4/Tanh:y:06residualblock_5/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_34/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_34/BiasAddBiasAdd)residualblock_5/dense_34/MatMul:product:07residualblock_5/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_5/add_5/addAddV2)residualblock_5/dense_33/BiasAdd:output:0)residualblock_5/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_5/TanhTanhresidualblock_5/add_5/add:z:0*
T0*'
_output_shapes
:?????????2?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_35/MatMulMatMulresidualblock_5/Tanh:y:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_35/ExpExpdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicerescaling_3/add:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlicerescaling_3/add:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerescaling_3/add:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????`
MulMuldense_35/Exp:y:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp.^residualblock/dense_17/BiasAdd/ReadVariableOp-^residualblock/dense_17/MatMul/ReadVariableOp.^residualblock/dense_18/BiasAdd/ReadVariableOp-^residualblock/dense_18/MatMul/ReadVariableOp.^residualblock/dense_19/BiasAdd/ReadVariableOp-^residualblock/dense_19/MatMul/ReadVariableOp0^residualblock_1/dense_20/BiasAdd/ReadVariableOp/^residualblock_1/dense_20/MatMul/ReadVariableOp0^residualblock_1/dense_21/BiasAdd/ReadVariableOp/^residualblock_1/dense_21/MatMul/ReadVariableOp0^residualblock_1/dense_22/BiasAdd/ReadVariableOp/^residualblock_1/dense_22/MatMul/ReadVariableOp0^residualblock_2/dense_23/BiasAdd/ReadVariableOp/^residualblock_2/dense_23/MatMul/ReadVariableOp0^residualblock_2/dense_24/BiasAdd/ReadVariableOp/^residualblock_2/dense_24/MatMul/ReadVariableOp0^residualblock_2/dense_25/BiasAdd/ReadVariableOp/^residualblock_2/dense_25/MatMul/ReadVariableOp0^residualblock_3/dense_26/BiasAdd/ReadVariableOp/^residualblock_3/dense_26/MatMul/ReadVariableOp0^residualblock_3/dense_27/BiasAdd/ReadVariableOp/^residualblock_3/dense_27/MatMul/ReadVariableOp0^residualblock_3/dense_28/BiasAdd/ReadVariableOp/^residualblock_3/dense_28/MatMul/ReadVariableOp0^residualblock_4/dense_29/BiasAdd/ReadVariableOp/^residualblock_4/dense_29/MatMul/ReadVariableOp0^residualblock_4/dense_30/BiasAdd/ReadVariableOp/^residualblock_4/dense_30/MatMul/ReadVariableOp0^residualblock_4/dense_31/BiasAdd/ReadVariableOp/^residualblock_4/dense_31/MatMul/ReadVariableOp0^residualblock_5/dense_32/BiasAdd/ReadVariableOp/^residualblock_5/dense_32/MatMul/ReadVariableOp0^residualblock_5/dense_33/BiasAdd/ReadVariableOp/^residualblock_5/dense_33/MatMul/ReadVariableOp0^residualblock_5/dense_34/BiasAdd/ReadVariableOp/^residualblock_5/dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2^
-residualblock/dense_17/BiasAdd/ReadVariableOp-residualblock/dense_17/BiasAdd/ReadVariableOp2\
,residualblock/dense_17/MatMul/ReadVariableOp,residualblock/dense_17/MatMul/ReadVariableOp2^
-residualblock/dense_18/BiasAdd/ReadVariableOp-residualblock/dense_18/BiasAdd/ReadVariableOp2\
,residualblock/dense_18/MatMul/ReadVariableOp,residualblock/dense_18/MatMul/ReadVariableOp2^
-residualblock/dense_19/BiasAdd/ReadVariableOp-residualblock/dense_19/BiasAdd/ReadVariableOp2\
,residualblock/dense_19/MatMul/ReadVariableOp,residualblock/dense_19/MatMul/ReadVariableOp2b
/residualblock_1/dense_20/BiasAdd/ReadVariableOp/residualblock_1/dense_20/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_20/MatMul/ReadVariableOp.residualblock_1/dense_20/MatMul/ReadVariableOp2b
/residualblock_1/dense_21/BiasAdd/ReadVariableOp/residualblock_1/dense_21/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_21/MatMul/ReadVariableOp.residualblock_1/dense_21/MatMul/ReadVariableOp2b
/residualblock_1/dense_22/BiasAdd/ReadVariableOp/residualblock_1/dense_22/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_22/MatMul/ReadVariableOp.residualblock_1/dense_22/MatMul/ReadVariableOp2b
/residualblock_2/dense_23/BiasAdd/ReadVariableOp/residualblock_2/dense_23/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_23/MatMul/ReadVariableOp.residualblock_2/dense_23/MatMul/ReadVariableOp2b
/residualblock_2/dense_24/BiasAdd/ReadVariableOp/residualblock_2/dense_24/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_24/MatMul/ReadVariableOp.residualblock_2/dense_24/MatMul/ReadVariableOp2b
/residualblock_2/dense_25/BiasAdd/ReadVariableOp/residualblock_2/dense_25/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_25/MatMul/ReadVariableOp.residualblock_2/dense_25/MatMul/ReadVariableOp2b
/residualblock_3/dense_26/BiasAdd/ReadVariableOp/residualblock_3/dense_26/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_26/MatMul/ReadVariableOp.residualblock_3/dense_26/MatMul/ReadVariableOp2b
/residualblock_3/dense_27/BiasAdd/ReadVariableOp/residualblock_3/dense_27/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_27/MatMul/ReadVariableOp.residualblock_3/dense_27/MatMul/ReadVariableOp2b
/residualblock_3/dense_28/BiasAdd/ReadVariableOp/residualblock_3/dense_28/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_28/MatMul/ReadVariableOp.residualblock_3/dense_28/MatMul/ReadVariableOp2b
/residualblock_4/dense_29/BiasAdd/ReadVariableOp/residualblock_4/dense_29/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_29/MatMul/ReadVariableOp.residualblock_4/dense_29/MatMul/ReadVariableOp2b
/residualblock_4/dense_30/BiasAdd/ReadVariableOp/residualblock_4/dense_30/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_30/MatMul/ReadVariableOp.residualblock_4/dense_30/MatMul/ReadVariableOp2b
/residualblock_4/dense_31/BiasAdd/ReadVariableOp/residualblock_4/dense_31/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_31/MatMul/ReadVariableOp.residualblock_4/dense_31/MatMul/ReadVariableOp2b
/residualblock_5/dense_32/BiasAdd/ReadVariableOp/residualblock_5/dense_32/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_32/MatMul/ReadVariableOp.residualblock_5/dense_32/MatMul/ReadVariableOp2b
/residualblock_5/dense_33/BiasAdd/ReadVariableOp/residualblock_5/dense_33/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_33/MatMul/ReadVariableOp.residualblock_5/dense_33/MatMul/ReadVariableOp2b
/residualblock_5/dense_34/BiasAdd/ReadVariableOp/residualblock_5/dense_34/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_34/MatMul/ReadVariableOp.residualblock_5/dense_34/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3152525

inputs9
'dense_26_matmul_readvariableop_resource:226
(dense_26_biasadd_readvariableop_resource:29
'dense_27_matmul_readvariableop_resource:226
(dense_27_biasadd_readvariableop_resource:29
'dense_28_matmul_readvariableop_resource:226
(dense_28_biasadd_readvariableop_resource:2
identity??dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_26/MatMulMatMulinputs&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_26/TanhTanhdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_27/MatMulMatMuldense_26/Tanh:y:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_28/MatMulMatMulinputs&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_3/addAddV2dense_27/BiasAdd:output:0dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_3/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809

inputs9
'dense_23_matmul_readvariableop_resource:226
(dense_23_biasadd_readvariableop_resource:29
'dense_24_matmul_readvariableop_resource:226
(dense_24_biasadd_readvariableop_resource:29
'dense_25_matmul_readvariableop_resource:226
(dense_25_biasadd_readvariableop_resource:2
identity??dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_24/MatMulMatMuldense_23/Tanh:y:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_2/addAddV2dense_24/BiasAdd:output:0dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_2/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?	
3__inference_travel_times_nn_3_layer_call_fn_3151999

inputs
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:22
	unknown_6:2
	unknown_7:22
	unknown_8:2
	unknown_9:22

unknown_10:2

unknown_11:22

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:22

unknown_16:2

unknown_17:22

unknown_18:2

unknown_19:22

unknown_20:2

unknown_21:22

unknown_22:2

unknown_23:22

unknown_24:2

unknown_25:22

unknown_26:2

unknown_27:22

unknown_28:2

unknown_29:22

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:22

unknown_34:2

unknown_35:2

unknown_36:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?	
3__inference_travel_times_nn_3_layer_call_fn_3151057
input_1
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:22
	unknown_6:2
	unknown_7:22
	unknown_8:2
	unknown_9:22

unknown_10:2

unknown_11:22

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:22

unknown_16:2

unknown_17:22

unknown_18:2

unknown_19:22

unknown_20:2

unknown_21:22

unknown_22:2

unknown_23:22

unknown_24:2

unknown_25:22

unknown_26:2

unknown_27:22

unknown_28:2

unknown_29:22

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:22

unknown_34:2

unknown_35:2

unknown_36:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3150978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?O
 __inference__traced_save_3153007
file_prefix@
<savev2_travel_times_nn_3_dense_35_kernel_read_readvariableop>
:savev2_travel_times_nn_3_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_dense_17_kernel_read_readvariableopL
Hsavev2_travel_times_nn_3_residualblock_dense_17_bias_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_dense_18_kernel_read_readvariableopL
Hsavev2_travel_times_nn_3_residualblock_dense_18_bias_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_dense_19_kernel_read_readvariableopL
Hsavev2_travel_times_nn_3_residualblock_dense_19_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_1_dense_20_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_1_dense_20_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_1_dense_21_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_1_dense_21_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_1_dense_22_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_1_dense_22_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_2_dense_23_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_2_dense_23_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_2_dense_24_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_2_dense_24_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_2_dense_25_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_2_dense_25_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_3_dense_26_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_3_dense_26_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_3_dense_27_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_3_dense_27_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_3_dense_28_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_3_dense_28_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_4_dense_29_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_4_dense_29_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_4_dense_30_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_4_dense_30_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_4_dense_31_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_4_dense_31_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_5_dense_32_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_5_dense_32_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_5_dense_33_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_5_dense_33_bias_read_readvariableopP
Lsavev2_travel_times_nn_3_residualblock_5_dense_34_kernel_read_readvariableopN
Jsavev2_travel_times_nn_3_residualblock_5_dense_34_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableopG
Csavev2_adam_travel_times_nn_3_dense_35_kernel_m_read_readvariableopE
Asavev2_adam_travel_times_nn_3_dense_35_bias_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_17_kernel_m_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_17_bias_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_18_kernel_m_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_18_bias_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_19_kernel_m_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_19_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_bias_m_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_m_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_bias_m_read_readvariableopG
Csavev2_adam_travel_times_nn_3_dense_35_kernel_v_read_readvariableopE
Asavev2_adam_travel_times_nn_3_dense_35_bias_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_17_kernel_v_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_17_bias_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_18_kernel_v_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_18_bias_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_dense_19_kernel_v_read_readvariableopS
Osavev2_adam_travel_times_nn_3_residualblock_dense_19_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_bias_v_read_readvariableopW
Ssavev2_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_v_read_readvariableopU
Qsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?:
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?9
value?9B?9~B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?M
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_travel_times_nn_3_dense_35_kernel_read_readvariableop:savev2_travel_times_nn_3_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopJsavev2_travel_times_nn_3_residualblock_dense_17_kernel_read_readvariableopHsavev2_travel_times_nn_3_residualblock_dense_17_bias_read_readvariableopJsavev2_travel_times_nn_3_residualblock_dense_18_kernel_read_readvariableopHsavev2_travel_times_nn_3_residualblock_dense_18_bias_read_readvariableopJsavev2_travel_times_nn_3_residualblock_dense_19_kernel_read_readvariableopHsavev2_travel_times_nn_3_residualblock_dense_19_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_1_dense_20_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_1_dense_20_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_1_dense_21_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_1_dense_21_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_1_dense_22_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_1_dense_22_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_2_dense_23_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_2_dense_23_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_2_dense_24_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_2_dense_24_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_2_dense_25_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_2_dense_25_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_3_dense_26_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_3_dense_26_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_3_dense_27_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_3_dense_27_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_3_dense_28_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_3_dense_28_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_4_dense_29_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_4_dense_29_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_4_dense_30_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_4_dense_30_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_4_dense_31_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_4_dense_31_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_5_dense_32_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_5_dense_32_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_5_dense_33_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_5_dense_33_bias_read_readvariableopLsavev2_travel_times_nn_3_residualblock_5_dense_34_kernel_read_readvariableopJsavev2_travel_times_nn_3_residualblock_5_dense_34_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopCsavev2_adam_travel_times_nn_3_dense_35_kernel_m_read_readvariableopAsavev2_adam_travel_times_nn_3_dense_35_bias_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_17_kernel_m_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_17_bias_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_18_kernel_m_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_18_bias_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_19_kernel_m_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_19_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_bias_m_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_m_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_bias_m_read_readvariableopCsavev2_adam_travel_times_nn_3_dense_35_kernel_v_read_readvariableopAsavev2_adam_travel_times_nn_3_dense_35_bias_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_17_kernel_v_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_17_bias_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_18_kernel_v_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_18_bias_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_dense_19_kernel_v_read_readvariableopOsavev2_adam_travel_times_nn_3_residualblock_dense_19_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_20_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_21_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_1_dense_22_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_23_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_24_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_2_dense_25_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_26_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_27_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_3_dense_28_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_29_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_30_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_4_dense_31_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_32_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_33_bias_v_read_readvariableopSsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_v_read_readvariableopQsavev2_adam_travel_times_nn_3_residualblock_5_dense_34_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2~	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:: : : : : :2:2:22:2:2:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2: : : : : : :2::2:2:22:2:2:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:2::2:2:22:2:2:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2:22:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 	

_output_shapes
:2:$
 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$  

_output_shapes

:22: !

_output_shapes
:2:$" 

_output_shapes

:22: #

_output_shapes
:2:$$ 

_output_shapes

:22: %

_output_shapes
:2:$& 

_output_shapes

:22: '

_output_shapes
:2:$( 

_output_shapes

:22: )

_output_shapes
:2:$* 

_output_shapes

:22: +

_output_shapes
:2:,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :$2 

_output_shapes

:2: 3

_output_shapes
::$4 

_output_shapes

:2: 5

_output_shapes
:2:$6 

_output_shapes

:22: 7

_output_shapes
:2:$8 

_output_shapes

:2: 9

_output_shapes
:2:$: 

_output_shapes

:22: ;

_output_shapes
:2:$< 

_output_shapes

:22: =

_output_shapes
:2:$> 

_output_shapes

:22: ?

_output_shapes
:2:$@ 

_output_shapes

:22: A

_output_shapes
:2:$B 

_output_shapes

:22: C

_output_shapes
:2:$D 

_output_shapes

:22: E

_output_shapes
:2:$F 

_output_shapes

:22: G

_output_shapes
:2:$H 

_output_shapes

:22: I

_output_shapes
:2:$J 

_output_shapes

:22: K

_output_shapes
:2:$L 

_output_shapes

:22: M

_output_shapes
:2:$N 

_output_shapes

:22: O

_output_shapes
:2:$P 

_output_shapes

:22: Q

_output_shapes
:2:$R 

_output_shapes

:22: S

_output_shapes
:2:$T 

_output_shapes

:22: U

_output_shapes
:2:$V 

_output_shapes

:22: W

_output_shapes
:2:$X 

_output_shapes

:2: Y

_output_shapes
::$Z 

_output_shapes

:2: [

_output_shapes
:2:$\ 

_output_shapes

:22: ]

_output_shapes
:2:$^ 

_output_shapes

:2: _

_output_shapes
:2:$` 

_output_shapes

:22: a

_output_shapes
:2:$b 

_output_shapes

:22: c

_output_shapes
:2:$d 

_output_shapes

:22: e

_output_shapes
:2:$f 

_output_shapes

:22: g

_output_shapes
:2:$h 

_output_shapes

:22: i

_output_shapes
:2:$j 

_output_shapes

:22: k

_output_shapes
:2:$l 

_output_shapes

:22: m

_output_shapes
:2:$n 

_output_shapes

:22: o

_output_shapes
:2:$p 

_output_shapes

:22: q

_output_shapes
:2:$r 

_output_shapes

:22: s

_output_shapes
:2:$t 

_output_shapes

:22: u

_output_shapes
:2:$v 

_output_shapes

:22: w

_output_shapes
:2:$x 

_output_shapes

:22: y

_output_shapes
:2:$z 

_output_shapes

:22: {

_output_shapes
:2:$| 

_output_shapes

:22: }

_output_shapes
:2:~

_output_shapes
: 
?
?
1__inference_residualblock_2_layer_call_fn_3152458

inputs
unknown:22
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:22
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
*__inference_dense_35_layer_call_fn_3152346

inputs
unknown:2
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
d
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3152337

inputs
identityO
Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2????Mb0?M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    N
mulMulinputsCast:y:0*
T0*'
_output_shapes
:?????????Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_residualblock_layer_call_and_return_conditional_losses_3152399

inputs9
'dense_17_matmul_readvariableop_resource:26
(dense_17_biasadd_readvariableop_resource:29
'dense_18_matmul_readvariableop_resource:226
(dense_18_biasadd_readvariableop_resource:29
'dense_19_matmul_readvariableop_resource:26
(dense_19_biasadd_readvariableop_resource:2
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0{
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_17/TanhTanhdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_18/MatMulMatMuldense_17/Tanh:y:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0{
dense_19/MatMulMatMulinputs&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2x
add/addAddV2dense_18/BiasAdd:output:0dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2K
TanhTanhadd/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151640
input_1'
residualblock_3151536:2#
residualblock_3151538:2'
residualblock_3151540:22#
residualblock_3151542:2'
residualblock_3151544:2#
residualblock_3151546:2)
residualblock_1_3151549:22%
residualblock_1_3151551:2)
residualblock_1_3151553:22%
residualblock_1_3151555:2)
residualblock_1_3151557:22%
residualblock_1_3151559:2)
residualblock_2_3151562:22%
residualblock_2_3151564:2)
residualblock_2_3151566:22%
residualblock_2_3151568:2)
residualblock_2_3151570:22%
residualblock_2_3151572:2)
residualblock_3_3151575:22%
residualblock_3_3151577:2)
residualblock_3_3151579:22%
residualblock_3_3151581:2)
residualblock_3_3151583:22%
residualblock_3_3151585:2)
residualblock_4_3151588:22%
residualblock_4_3151590:2)
residualblock_4_3151592:22%
residualblock_4_3151594:2)
residualblock_4_3151596:22%
residualblock_4_3151598:2)
residualblock_5_3151601:22%
residualblock_5_3151603:2)
residualblock_5_3151605:22%
residualblock_5_3151607:2)
residualblock_5_3151609:22%
residualblock_5_3151611:2"
dense_35_3151614:2
dense_35_3151616:
identity?? dense_35/StatefulPartitionedCall?%residualblock/StatefulPartitionedCall?'residualblock_1/StatefulPartitionedCall?'residualblock_2/StatefulPartitionedCall?'residualblock_3/StatefulPartitionedCall?'residualblock_4/StatefulPartitionedCall?'residualblock_5/StatefulPartitionedCall?
rescaling_3/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704?
%residualblock/StatefulPartitionedCallStatefulPartitionedCall$rescaling_3/PartitionedCall:output:0residualblock_3151536residualblock_3151538residualblock_3151540residualblock_3151542residualblock_3151544residualblock_3151546*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731?
'residualblock_1/StatefulPartitionedCallStatefulPartitionedCall.residualblock/StatefulPartitionedCall:output:0residualblock_1_3151549residualblock_1_3151551residualblock_1_3151553residualblock_1_3151555residualblock_1_3151557residualblock_1_3151559*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770?
'residualblock_2/StatefulPartitionedCallStatefulPartitionedCall0residualblock_1/StatefulPartitionedCall:output:0residualblock_2_3151562residualblock_2_3151564residualblock_2_3151566residualblock_2_3151568residualblock_2_3151570residualblock_2_3151572*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809?
'residualblock_3/StatefulPartitionedCallStatefulPartitionedCall0residualblock_2/StatefulPartitionedCall:output:0residualblock_3_3151575residualblock_3_3151577residualblock_3_3151579residualblock_3_3151581residualblock_3_3151583residualblock_3_3151585*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848?
'residualblock_4/StatefulPartitionedCallStatefulPartitionedCall0residualblock_3/StatefulPartitionedCall:output:0residualblock_4_3151588residualblock_4_3151590residualblock_4_3151592residualblock_4_3151594residualblock_4_3151596residualblock_4_3151598*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887?
'residualblock_5/StatefulPartitionedCallStatefulPartitionedCall0residualblock_4/StatefulPartitionedCall:output:0residualblock_5_3151601residualblock_5_3151603residualblock_5_3151605residualblock_5_3151607residualblock_5_3151609residualblock_5_3151611*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall0residualblock_5/StatefulPartitionedCall:output:0dense_35_3151614dense_35_3151616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlice$rescaling_3/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
MulMul)dense_35/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall&^residualblock/StatefulPartitionedCall(^residualblock_1/StatefulPartitionedCall(^residualblock_2/StatefulPartitionedCall(^residualblock_3/StatefulPartitionedCall(^residualblock_4/StatefulPartitionedCall(^residualblock_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%residualblock/StatefulPartitionedCall%residualblock/StatefulPartitionedCall2R
'residualblock_1/StatefulPartitionedCall'residualblock_1/StatefulPartitionedCall2R
'residualblock_2/StatefulPartitionedCall'residualblock_2/StatefulPartitionedCall2R
'residualblock_3/StatefulPartitionedCall'residualblock_3/StatefulPartitionedCall2R
'residualblock_4/StatefulPartitionedCall'residualblock_4/StatefulPartitionedCall2R
'residualblock_5/StatefulPartitionedCall'residualblock_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
d
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704

inputs
identityO
Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2????Mb0?M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    N
mulMulinputsCast:y:0*
T0*'
_output_shapes
:?????????Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?i
#__inference__traced_restore_3153392
file_prefixD
2assignvariableop_travel_times_nn_3_dense_35_kernel:2@
2assignvariableop_1_travel_times_nn_3_dense_35_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: T
Bassignvariableop_7_travel_times_nn_3_residualblock_dense_17_kernel:2N
@assignvariableop_8_travel_times_nn_3_residualblock_dense_17_bias:2T
Bassignvariableop_9_travel_times_nn_3_residualblock_dense_18_kernel:22O
Aassignvariableop_10_travel_times_nn_3_residualblock_dense_18_bias:2U
Cassignvariableop_11_travel_times_nn_3_residualblock_dense_19_kernel:2O
Aassignvariableop_12_travel_times_nn_3_residualblock_dense_19_bias:2W
Eassignvariableop_13_travel_times_nn_3_residualblock_1_dense_20_kernel:22Q
Cassignvariableop_14_travel_times_nn_3_residualblock_1_dense_20_bias:2W
Eassignvariableop_15_travel_times_nn_3_residualblock_1_dense_21_kernel:22Q
Cassignvariableop_16_travel_times_nn_3_residualblock_1_dense_21_bias:2W
Eassignvariableop_17_travel_times_nn_3_residualblock_1_dense_22_kernel:22Q
Cassignvariableop_18_travel_times_nn_3_residualblock_1_dense_22_bias:2W
Eassignvariableop_19_travel_times_nn_3_residualblock_2_dense_23_kernel:22Q
Cassignvariableop_20_travel_times_nn_3_residualblock_2_dense_23_bias:2W
Eassignvariableop_21_travel_times_nn_3_residualblock_2_dense_24_kernel:22Q
Cassignvariableop_22_travel_times_nn_3_residualblock_2_dense_24_bias:2W
Eassignvariableop_23_travel_times_nn_3_residualblock_2_dense_25_kernel:22Q
Cassignvariableop_24_travel_times_nn_3_residualblock_2_dense_25_bias:2W
Eassignvariableop_25_travel_times_nn_3_residualblock_3_dense_26_kernel:22Q
Cassignvariableop_26_travel_times_nn_3_residualblock_3_dense_26_bias:2W
Eassignvariableop_27_travel_times_nn_3_residualblock_3_dense_27_kernel:22Q
Cassignvariableop_28_travel_times_nn_3_residualblock_3_dense_27_bias:2W
Eassignvariableop_29_travel_times_nn_3_residualblock_3_dense_28_kernel:22Q
Cassignvariableop_30_travel_times_nn_3_residualblock_3_dense_28_bias:2W
Eassignvariableop_31_travel_times_nn_3_residualblock_4_dense_29_kernel:22Q
Cassignvariableop_32_travel_times_nn_3_residualblock_4_dense_29_bias:2W
Eassignvariableop_33_travel_times_nn_3_residualblock_4_dense_30_kernel:22Q
Cassignvariableop_34_travel_times_nn_3_residualblock_4_dense_30_bias:2W
Eassignvariableop_35_travel_times_nn_3_residualblock_4_dense_31_kernel:22Q
Cassignvariableop_36_travel_times_nn_3_residualblock_4_dense_31_bias:2W
Eassignvariableop_37_travel_times_nn_3_residualblock_5_dense_32_kernel:22Q
Cassignvariableop_38_travel_times_nn_3_residualblock_5_dense_32_bias:2W
Eassignvariableop_39_travel_times_nn_3_residualblock_5_dense_33_kernel:22Q
Cassignvariableop_40_travel_times_nn_3_residualblock_5_dense_33_bias:2W
Eassignvariableop_41_travel_times_nn_3_residualblock_5_dense_34_kernel:22Q
Cassignvariableop_42_travel_times_nn_3_residualblock_5_dense_34_bias:2#
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: %
assignvariableop_47_total_2: %
assignvariableop_48_count_2: N
<assignvariableop_49_adam_travel_times_nn_3_dense_35_kernel_m:2H
:assignvariableop_50_adam_travel_times_nn_3_dense_35_bias_m:\
Jassignvariableop_51_adam_travel_times_nn_3_residualblock_dense_17_kernel_m:2V
Hassignvariableop_52_adam_travel_times_nn_3_residualblock_dense_17_bias_m:2\
Jassignvariableop_53_adam_travel_times_nn_3_residualblock_dense_18_kernel_m:22V
Hassignvariableop_54_adam_travel_times_nn_3_residualblock_dense_18_bias_m:2\
Jassignvariableop_55_adam_travel_times_nn_3_residualblock_dense_19_kernel_m:2V
Hassignvariableop_56_adam_travel_times_nn_3_residualblock_dense_19_bias_m:2^
Lassignvariableop_57_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_m:22X
Jassignvariableop_58_adam_travel_times_nn_3_residualblock_1_dense_20_bias_m:2^
Lassignvariableop_59_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_m:22X
Jassignvariableop_60_adam_travel_times_nn_3_residualblock_1_dense_21_bias_m:2^
Lassignvariableop_61_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_m:22X
Jassignvariableop_62_adam_travel_times_nn_3_residualblock_1_dense_22_bias_m:2^
Lassignvariableop_63_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_m:22X
Jassignvariableop_64_adam_travel_times_nn_3_residualblock_2_dense_23_bias_m:2^
Lassignvariableop_65_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_m:22X
Jassignvariableop_66_adam_travel_times_nn_3_residualblock_2_dense_24_bias_m:2^
Lassignvariableop_67_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_m:22X
Jassignvariableop_68_adam_travel_times_nn_3_residualblock_2_dense_25_bias_m:2^
Lassignvariableop_69_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_m:22X
Jassignvariableop_70_adam_travel_times_nn_3_residualblock_3_dense_26_bias_m:2^
Lassignvariableop_71_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_m:22X
Jassignvariableop_72_adam_travel_times_nn_3_residualblock_3_dense_27_bias_m:2^
Lassignvariableop_73_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_m:22X
Jassignvariableop_74_adam_travel_times_nn_3_residualblock_3_dense_28_bias_m:2^
Lassignvariableop_75_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_m:22X
Jassignvariableop_76_adam_travel_times_nn_3_residualblock_4_dense_29_bias_m:2^
Lassignvariableop_77_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_m:22X
Jassignvariableop_78_adam_travel_times_nn_3_residualblock_4_dense_30_bias_m:2^
Lassignvariableop_79_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_m:22X
Jassignvariableop_80_adam_travel_times_nn_3_residualblock_4_dense_31_bias_m:2^
Lassignvariableop_81_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_m:22X
Jassignvariableop_82_adam_travel_times_nn_3_residualblock_5_dense_32_bias_m:2^
Lassignvariableop_83_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_m:22X
Jassignvariableop_84_adam_travel_times_nn_3_residualblock_5_dense_33_bias_m:2^
Lassignvariableop_85_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_m:22X
Jassignvariableop_86_adam_travel_times_nn_3_residualblock_5_dense_34_bias_m:2N
<assignvariableop_87_adam_travel_times_nn_3_dense_35_kernel_v:2H
:assignvariableop_88_adam_travel_times_nn_3_dense_35_bias_v:\
Jassignvariableop_89_adam_travel_times_nn_3_residualblock_dense_17_kernel_v:2V
Hassignvariableop_90_adam_travel_times_nn_3_residualblock_dense_17_bias_v:2\
Jassignvariableop_91_adam_travel_times_nn_3_residualblock_dense_18_kernel_v:22V
Hassignvariableop_92_adam_travel_times_nn_3_residualblock_dense_18_bias_v:2\
Jassignvariableop_93_adam_travel_times_nn_3_residualblock_dense_19_kernel_v:2V
Hassignvariableop_94_adam_travel_times_nn_3_residualblock_dense_19_bias_v:2^
Lassignvariableop_95_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_v:22X
Jassignvariableop_96_adam_travel_times_nn_3_residualblock_1_dense_20_bias_v:2^
Lassignvariableop_97_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_v:22X
Jassignvariableop_98_adam_travel_times_nn_3_residualblock_1_dense_21_bias_v:2^
Lassignvariableop_99_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_v:22Y
Kassignvariableop_100_adam_travel_times_nn_3_residualblock_1_dense_22_bias_v:2_
Massignvariableop_101_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_v:22Y
Kassignvariableop_102_adam_travel_times_nn_3_residualblock_2_dense_23_bias_v:2_
Massignvariableop_103_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_v:22Y
Kassignvariableop_104_adam_travel_times_nn_3_residualblock_2_dense_24_bias_v:2_
Massignvariableop_105_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_v:22Y
Kassignvariableop_106_adam_travel_times_nn_3_residualblock_2_dense_25_bias_v:2_
Massignvariableop_107_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_v:22Y
Kassignvariableop_108_adam_travel_times_nn_3_residualblock_3_dense_26_bias_v:2_
Massignvariableop_109_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_v:22Y
Kassignvariableop_110_adam_travel_times_nn_3_residualblock_3_dense_27_bias_v:2_
Massignvariableop_111_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_v:22Y
Kassignvariableop_112_adam_travel_times_nn_3_residualblock_3_dense_28_bias_v:2_
Massignvariableop_113_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_v:22Y
Kassignvariableop_114_adam_travel_times_nn_3_residualblock_4_dense_29_bias_v:2_
Massignvariableop_115_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_v:22Y
Kassignvariableop_116_adam_travel_times_nn_3_residualblock_4_dense_30_bias_v:2_
Massignvariableop_117_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_v:22Y
Kassignvariableop_118_adam_travel_times_nn_3_residualblock_4_dense_31_bias_v:2_
Massignvariableop_119_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_v:22Y
Kassignvariableop_120_adam_travel_times_nn_3_residualblock_5_dense_32_bias_v:2_
Massignvariableop_121_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_v:22Y
Kassignvariableop_122_adam_travel_times_nn_3_residualblock_5_dense_33_bias_v:2_
Massignvariableop_123_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_v:22Y
Kassignvariableop_124_adam_travel_times_nn_3_residualblock_5_dense_34_bias_v:2
identity_126??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?:
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?9
value?9B?9~B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2~	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp2assignvariableop_travel_times_nn_3_dense_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp2assignvariableop_1_travel_times_nn_3_dense_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpBassignvariableop_7_travel_times_nn_3_residualblock_dense_17_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp@assignvariableop_8_travel_times_nn_3_residualblock_dense_17_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpBassignvariableop_9_travel_times_nn_3_residualblock_dense_18_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpAassignvariableop_10_travel_times_nn_3_residualblock_dense_18_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_travel_times_nn_3_residualblock_dense_19_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpAassignvariableop_12_travel_times_nn_3_residualblock_dense_19_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpEassignvariableop_13_travel_times_nn_3_residualblock_1_dense_20_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpCassignvariableop_14_travel_times_nn_3_residualblock_1_dense_20_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpEassignvariableop_15_travel_times_nn_3_residualblock_1_dense_21_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpCassignvariableop_16_travel_times_nn_3_residualblock_1_dense_21_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_travel_times_nn_3_residualblock_1_dense_22_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpCassignvariableop_18_travel_times_nn_3_residualblock_1_dense_22_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpEassignvariableop_19_travel_times_nn_3_residualblock_2_dense_23_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpCassignvariableop_20_travel_times_nn_3_residualblock_2_dense_23_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpEassignvariableop_21_travel_times_nn_3_residualblock_2_dense_24_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpCassignvariableop_22_travel_times_nn_3_residualblock_2_dense_24_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpEassignvariableop_23_travel_times_nn_3_residualblock_2_dense_25_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpCassignvariableop_24_travel_times_nn_3_residualblock_2_dense_25_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpEassignvariableop_25_travel_times_nn_3_residualblock_3_dense_26_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_travel_times_nn_3_residualblock_3_dense_26_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpEassignvariableop_27_travel_times_nn_3_residualblock_3_dense_27_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpCassignvariableop_28_travel_times_nn_3_residualblock_3_dense_27_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpEassignvariableop_29_travel_times_nn_3_residualblock_3_dense_28_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpCassignvariableop_30_travel_times_nn_3_residualblock_3_dense_28_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpEassignvariableop_31_travel_times_nn_3_residualblock_4_dense_29_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpCassignvariableop_32_travel_times_nn_3_residualblock_4_dense_29_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpEassignvariableop_33_travel_times_nn_3_residualblock_4_dense_30_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpCassignvariableop_34_travel_times_nn_3_residualblock_4_dense_30_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpEassignvariableop_35_travel_times_nn_3_residualblock_4_dense_31_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpCassignvariableop_36_travel_times_nn_3_residualblock_4_dense_31_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpEassignvariableop_37_travel_times_nn_3_residualblock_5_dense_32_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpCassignvariableop_38_travel_times_nn_3_residualblock_5_dense_32_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpEassignvariableop_39_travel_times_nn_3_residualblock_5_dense_33_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpCassignvariableop_40_travel_times_nn_3_residualblock_5_dense_33_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpEassignvariableop_41_travel_times_nn_3_residualblock_5_dense_34_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpCassignvariableop_42_travel_times_nn_3_residualblock_5_dense_34_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp<assignvariableop_49_adam_travel_times_nn_3_dense_35_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp:assignvariableop_50_adam_travel_times_nn_3_dense_35_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpJassignvariableop_51_adam_travel_times_nn_3_residualblock_dense_17_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpHassignvariableop_52_adam_travel_times_nn_3_residualblock_dense_17_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpJassignvariableop_53_adam_travel_times_nn_3_residualblock_dense_18_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpHassignvariableop_54_adam_travel_times_nn_3_residualblock_dense_18_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpJassignvariableop_55_adam_travel_times_nn_3_residualblock_dense_19_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpHassignvariableop_56_adam_travel_times_nn_3_residualblock_dense_19_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpLassignvariableop_57_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpJassignvariableop_58_adam_travel_times_nn_3_residualblock_1_dense_20_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpLassignvariableop_59_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpJassignvariableop_60_adam_travel_times_nn_3_residualblock_1_dense_21_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpLassignvariableop_61_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpJassignvariableop_62_adam_travel_times_nn_3_residualblock_1_dense_22_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpJassignvariableop_64_adam_travel_times_nn_3_residualblock_2_dense_23_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpLassignvariableop_65_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpJassignvariableop_66_adam_travel_times_nn_3_residualblock_2_dense_24_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpLassignvariableop_67_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpJassignvariableop_68_adam_travel_times_nn_3_residualblock_2_dense_25_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpLassignvariableop_69_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpJassignvariableop_70_adam_travel_times_nn_3_residualblock_3_dense_26_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpLassignvariableop_71_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpJassignvariableop_72_adam_travel_times_nn_3_residualblock_3_dense_27_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpLassignvariableop_73_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpJassignvariableop_74_adam_travel_times_nn_3_residualblock_3_dense_28_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOpLassignvariableop_75_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpJassignvariableop_76_adam_travel_times_nn_3_residualblock_4_dense_29_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpLassignvariableop_77_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOpJassignvariableop_78_adam_travel_times_nn_3_residualblock_4_dense_30_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOpLassignvariableop_79_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOpJassignvariableop_80_adam_travel_times_nn_3_residualblock_4_dense_31_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOpLassignvariableop_81_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOpJassignvariableop_82_adam_travel_times_nn_3_residualblock_5_dense_32_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOpLassignvariableop_83_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOpJassignvariableop_84_adam_travel_times_nn_3_residualblock_5_dense_33_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOpLassignvariableop_85_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOpJassignvariableop_86_adam_travel_times_nn_3_residualblock_5_dense_34_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp<assignvariableop_87_adam_travel_times_nn_3_dense_35_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp:assignvariableop_88_adam_travel_times_nn_3_dense_35_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOpJassignvariableop_89_adam_travel_times_nn_3_residualblock_dense_17_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOpHassignvariableop_90_adam_travel_times_nn_3_residualblock_dense_17_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOpJassignvariableop_91_adam_travel_times_nn_3_residualblock_dense_18_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOpHassignvariableop_92_adam_travel_times_nn_3_residualblock_dense_18_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOpJassignvariableop_93_adam_travel_times_nn_3_residualblock_dense_19_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOpHassignvariableop_94_adam_travel_times_nn_3_residualblock_dense_19_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOpLassignvariableop_95_adam_travel_times_nn_3_residualblock_1_dense_20_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOpJassignvariableop_96_adam_travel_times_nn_3_residualblock_1_dense_20_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOpLassignvariableop_97_adam_travel_times_nn_3_residualblock_1_dense_21_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOpJassignvariableop_98_adam_travel_times_nn_3_residualblock_1_dense_21_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOpLassignvariableop_99_adam_travel_times_nn_3_residualblock_1_dense_22_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOpKassignvariableop_100_adam_travel_times_nn_3_residualblock_1_dense_22_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOpMassignvariableop_101_adam_travel_times_nn_3_residualblock_2_dense_23_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOpKassignvariableop_102_adam_travel_times_nn_3_residualblock_2_dense_23_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOpMassignvariableop_103_adam_travel_times_nn_3_residualblock_2_dense_24_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOpKassignvariableop_104_adam_travel_times_nn_3_residualblock_2_dense_24_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOpMassignvariableop_105_adam_travel_times_nn_3_residualblock_2_dense_25_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOpKassignvariableop_106_adam_travel_times_nn_3_residualblock_2_dense_25_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOpMassignvariableop_107_adam_travel_times_nn_3_residualblock_3_dense_26_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOpKassignvariableop_108_adam_travel_times_nn_3_residualblock_3_dense_26_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOpMassignvariableop_109_adam_travel_times_nn_3_residualblock_3_dense_27_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOpKassignvariableop_110_adam_travel_times_nn_3_residualblock_3_dense_27_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOpMassignvariableop_111_adam_travel_times_nn_3_residualblock_3_dense_28_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOpKassignvariableop_112_adam_travel_times_nn_3_residualblock_3_dense_28_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOpMassignvariableop_113_adam_travel_times_nn_3_residualblock_4_dense_29_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOpKassignvariableop_114_adam_travel_times_nn_3_residualblock_4_dense_29_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOpMassignvariableop_115_adam_travel_times_nn_3_residualblock_4_dense_30_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOpKassignvariableop_116_adam_travel_times_nn_3_residualblock_4_dense_30_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOpMassignvariableop_117_adam_travel_times_nn_3_residualblock_4_dense_31_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOpKassignvariableop_118_adam_travel_times_nn_3_residualblock_4_dense_31_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOpMassignvariableop_119_adam_travel_times_nn_3_residualblock_5_dense_32_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOpKassignvariableop_120_adam_travel_times_nn_3_residualblock_5_dense_32_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOpMassignvariableop_121_adam_travel_times_nn_3_residualblock_5_dense_33_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOpKassignvariableop_122_adam_travel_times_nn_3_residualblock_5_dense_33_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOpMassignvariableop_123_adam_travel_times_nn_3_residualblock_5_dense_34_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOpKassignvariableop_124_adam_travel_times_nn_3_residualblock_5_dense_34_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_125Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_126IdentityIdentity_125:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_126Identity_126:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
/__inference_residualblock_layer_call_fn_3152374

inputs
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????N
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityExp:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?L
?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151748
input_1'
residualblock_3151644:2#
residualblock_3151646:2'
residualblock_3151648:22#
residualblock_3151650:2'
residualblock_3151652:2#
residualblock_3151654:2)
residualblock_1_3151657:22%
residualblock_1_3151659:2)
residualblock_1_3151661:22%
residualblock_1_3151663:2)
residualblock_1_3151665:22%
residualblock_1_3151667:2)
residualblock_2_3151670:22%
residualblock_2_3151672:2)
residualblock_2_3151674:22%
residualblock_2_3151676:2)
residualblock_2_3151678:22%
residualblock_2_3151680:2)
residualblock_3_3151683:22%
residualblock_3_3151685:2)
residualblock_3_3151687:22%
residualblock_3_3151689:2)
residualblock_3_3151691:22%
residualblock_3_3151693:2)
residualblock_4_3151696:22%
residualblock_4_3151698:2)
residualblock_4_3151700:22%
residualblock_4_3151702:2)
residualblock_4_3151704:22%
residualblock_4_3151706:2)
residualblock_5_3151709:22%
residualblock_5_3151711:2)
residualblock_5_3151713:22%
residualblock_5_3151715:2)
residualblock_5_3151717:22%
residualblock_5_3151719:2"
dense_35_3151722:2
dense_35_3151724:
identity?? dense_35/StatefulPartitionedCall?%residualblock/StatefulPartitionedCall?'residualblock_1/StatefulPartitionedCall?'residualblock_2/StatefulPartitionedCall?'residualblock_3/StatefulPartitionedCall?'residualblock_4/StatefulPartitionedCall?'residualblock_5/StatefulPartitionedCall?
rescaling_3/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704?
%residualblock/StatefulPartitionedCallStatefulPartitionedCall$rescaling_3/PartitionedCall:output:0residualblock_3151644residualblock_3151646residualblock_3151648residualblock_3151650residualblock_3151652residualblock_3151654*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731?
'residualblock_1/StatefulPartitionedCallStatefulPartitionedCall.residualblock/StatefulPartitionedCall:output:0residualblock_1_3151657residualblock_1_3151659residualblock_1_3151661residualblock_1_3151663residualblock_1_3151665residualblock_1_3151667*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770?
'residualblock_2/StatefulPartitionedCallStatefulPartitionedCall0residualblock_1/StatefulPartitionedCall:output:0residualblock_2_3151670residualblock_2_3151672residualblock_2_3151674residualblock_2_3151676residualblock_2_3151678residualblock_2_3151680*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809?
'residualblock_3/StatefulPartitionedCallStatefulPartitionedCall0residualblock_2/StatefulPartitionedCall:output:0residualblock_3_3151683residualblock_3_3151685residualblock_3_3151687residualblock_3_3151689residualblock_3_3151691residualblock_3_3151693*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848?
'residualblock_4/StatefulPartitionedCallStatefulPartitionedCall0residualblock_3/StatefulPartitionedCall:output:0residualblock_4_3151696residualblock_4_3151698residualblock_4_3151700residualblock_4_3151702residualblock_4_3151704residualblock_4_3151706*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887?
'residualblock_5/StatefulPartitionedCallStatefulPartitionedCall0residualblock_4/StatefulPartitionedCall:output:0residualblock_5_3151709residualblock_5_3151711residualblock_5_3151713residualblock_5_3151715residualblock_5_3151717residualblock_5_3151719*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall0residualblock_5/StatefulPartitionedCall:output:0dense_35_3151722dense_35_3151724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlice$rescaling_3/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
MulMul)dense_35/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall&^residualblock/StatefulPartitionedCall(^residualblock_1/StatefulPartitionedCall(^residualblock_2/StatefulPartitionedCall(^residualblock_3/StatefulPartitionedCall(^residualblock_4/StatefulPartitionedCall(^residualblock_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%residualblock/StatefulPartitionedCall%residualblock/StatefulPartitionedCall2R
'residualblock_1/StatefulPartitionedCall'residualblock_1/StatefulPartitionedCall2R
'residualblock_2/StatefulPartitionedCall'residualblock_2/StatefulPartitionedCall2R
'residualblock_3/StatefulPartitionedCall'residualblock_3/StatefulPartitionedCall2R
'residualblock_4/StatefulPartitionedCall'residualblock_4/StatefulPartitionedCall2R
'residualblock_5/StatefulPartitionedCall'residualblock_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
E__inference_dense_35_layer_call_and_return_conditional_losses_3152357

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????N
ExpExpBiasAdd:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityExp:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
I
-__inference_rescaling_3_layer_call_fn_3152328

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3152441

inputs9
'dense_20_matmul_readvariableop_resource:226
(dense_20_biasadd_readvariableop_resource:29
'dense_21_matmul_readvariableop_resource:226
(dense_21_biasadd_readvariableop_resource:29
'dense_22_matmul_readvariableop_resource:226
(dense_22_biasadd_readvariableop_resource:2
identity??dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_21/MatMulMatMuldense_20/Tanh:y:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_22/MatMulMatMulinputs&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_1/addAddV2dense_21/BiasAdd:output:0dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_1/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?L
?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3150978

inputs'
residualblock_3150732:2#
residualblock_3150734:2'
residualblock_3150736:22#
residualblock_3150738:2'
residualblock_3150740:2#
residualblock_3150742:2)
residualblock_1_3150771:22%
residualblock_1_3150773:2)
residualblock_1_3150775:22%
residualblock_1_3150777:2)
residualblock_1_3150779:22%
residualblock_1_3150781:2)
residualblock_2_3150810:22%
residualblock_2_3150812:2)
residualblock_2_3150814:22%
residualblock_2_3150816:2)
residualblock_2_3150818:22%
residualblock_2_3150820:2)
residualblock_3_3150849:22%
residualblock_3_3150851:2)
residualblock_3_3150853:22%
residualblock_3_3150855:2)
residualblock_3_3150857:22%
residualblock_3_3150859:2)
residualblock_4_3150888:22%
residualblock_4_3150890:2)
residualblock_4_3150892:22%
residualblock_4_3150894:2)
residualblock_4_3150896:22%
residualblock_4_3150898:2)
residualblock_5_3150927:22%
residualblock_5_3150929:2)
residualblock_5_3150931:22%
residualblock_5_3150933:2)
residualblock_5_3150935:22%
residualblock_5_3150937:2"
dense_35_3150952:2
dense_35_3150954:
identity?? dense_35/StatefulPartitionedCall?%residualblock/StatefulPartitionedCall?'residualblock_1/StatefulPartitionedCall?'residualblock_2/StatefulPartitionedCall?'residualblock_3/StatefulPartitionedCall?'residualblock_4/StatefulPartitionedCall?'residualblock_5/StatefulPartitionedCall?
rescaling_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704?
%residualblock/StatefulPartitionedCallStatefulPartitionedCall$rescaling_3/PartitionedCall:output:0residualblock_3150732residualblock_3150734residualblock_3150736residualblock_3150738residualblock_3150740residualblock_3150742*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731?
'residualblock_1/StatefulPartitionedCallStatefulPartitionedCall.residualblock/StatefulPartitionedCall:output:0residualblock_1_3150771residualblock_1_3150773residualblock_1_3150775residualblock_1_3150777residualblock_1_3150779residualblock_1_3150781*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770?
'residualblock_2/StatefulPartitionedCallStatefulPartitionedCall0residualblock_1/StatefulPartitionedCall:output:0residualblock_2_3150810residualblock_2_3150812residualblock_2_3150814residualblock_2_3150816residualblock_2_3150818residualblock_2_3150820*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809?
'residualblock_3/StatefulPartitionedCallStatefulPartitionedCall0residualblock_2/StatefulPartitionedCall:output:0residualblock_3_3150849residualblock_3_3150851residualblock_3_3150853residualblock_3_3150855residualblock_3_3150857residualblock_3_3150859*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848?
'residualblock_4/StatefulPartitionedCallStatefulPartitionedCall0residualblock_3/StatefulPartitionedCall:output:0residualblock_4_3150888residualblock_4_3150890residualblock_4_3150892residualblock_4_3150894residualblock_4_3150896residualblock_4_3150898*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887?
'residualblock_5/StatefulPartitionedCallStatefulPartitionedCall0residualblock_4/StatefulPartitionedCall:output:0residualblock_5_3150927residualblock_5_3150929residualblock_5_3150931residualblock_5_3150933residualblock_5_3150935residualblock_5_3150937*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall0residualblock_5/StatefulPartitionedCall:output:0dense_35_3150952dense_35_3150954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlice$rescaling_3/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
MulMul)dense_35/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall&^residualblock/StatefulPartitionedCall(^residualblock_1/StatefulPartitionedCall(^residualblock_2/StatefulPartitionedCall(^residualblock_3/StatefulPartitionedCall(^residualblock_4/StatefulPartitionedCall(^residualblock_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%residualblock/StatefulPartitionedCall%residualblock/StatefulPartitionedCall2R
'residualblock_1/StatefulPartitionedCall'residualblock_1/StatefulPartitionedCall2R
'residualblock_2/StatefulPartitionedCall'residualblock_2/StatefulPartitionedCall2R
'residualblock_3/StatefulPartitionedCall'residualblock_3/StatefulPartitionedCall2R
'residualblock_4/StatefulPartitionedCall'residualblock_4/StatefulPartitionedCall2R
'residualblock_5/StatefulPartitionedCall'residualblock_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3152483

inputs9
'dense_23_matmul_readvariableop_resource:226
(dense_23_biasadd_readvariableop_resource:29
'dense_24_matmul_readvariableop_resource:226
(dense_24_biasadd_readvariableop_resource:29
'dense_25_matmul_readvariableop_resource:226
(dense_25_biasadd_readvariableop_resource:2
identity??dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_23/TanhTanhdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_24/MatMulMatMuldense_23/Tanh:y:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_2/addAddV2dense_24/BiasAdd:output:0dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_2/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_3151837
input_1
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:22
	unknown_6:2
	unknown_7:22
	unknown_8:2
	unknown_9:22

unknown_10:2

unknown_11:22

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:22

unknown_16:2

unknown_17:22

unknown_18:2

unknown_19:22

unknown_20:2

unknown_21:22

unknown_22:2

unknown_23:22

unknown_24:2

unknown_25:22

unknown_26:2

unknown_27:22

unknown_28:2

unknown_29:22

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:22

unknown_34:2

unknown_35:2

unknown_36:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_3150688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926

inputs9
'dense_32_matmul_readvariableop_resource:226
(dense_32_biasadd_readvariableop_resource:29
'dense_33_matmul_readvariableop_resource:226
(dense_33_biasadd_readvariableop_resource:29
'dense_34_matmul_readvariableop_resource:226
(dense_34_biasadd_readvariableop_resource:2
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_32/MatMulMatMulinputs&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_32/TanhTanhdense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_33/MatMulMatMuldense_32/Tanh:y:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_34/MatMulMatMulinputs&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_5/addAddV2dense_33/BiasAdd:output:0dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_5/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?L
?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151372

inputs'
residualblock_3151268:2#
residualblock_3151270:2'
residualblock_3151272:22#
residualblock_3151274:2'
residualblock_3151276:2#
residualblock_3151278:2)
residualblock_1_3151281:22%
residualblock_1_3151283:2)
residualblock_1_3151285:22%
residualblock_1_3151287:2)
residualblock_1_3151289:22%
residualblock_1_3151291:2)
residualblock_2_3151294:22%
residualblock_2_3151296:2)
residualblock_2_3151298:22%
residualblock_2_3151300:2)
residualblock_2_3151302:22%
residualblock_2_3151304:2)
residualblock_3_3151307:22%
residualblock_3_3151309:2)
residualblock_3_3151311:22%
residualblock_3_3151313:2)
residualblock_3_3151315:22%
residualblock_3_3151317:2)
residualblock_4_3151320:22%
residualblock_4_3151322:2)
residualblock_4_3151324:22%
residualblock_4_3151326:2)
residualblock_4_3151328:22%
residualblock_4_3151330:2)
residualblock_5_3151333:22%
residualblock_5_3151335:2)
residualblock_5_3151337:22%
residualblock_5_3151339:2)
residualblock_5_3151341:22%
residualblock_5_3151343:2"
dense_35_3151346:2
dense_35_3151348:
identity?? dense_35/StatefulPartitionedCall?%residualblock/StatefulPartitionedCall?'residualblock_1/StatefulPartitionedCall?'residualblock_2/StatefulPartitionedCall?'residualblock_3/StatefulPartitionedCall?'residualblock_4/StatefulPartitionedCall?'residualblock_5/StatefulPartitionedCall?
rescaling_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3150704?
%residualblock/StatefulPartitionedCallStatefulPartitionedCall$rescaling_3/PartitionedCall:output:0residualblock_3151268residualblock_3151270residualblock_3151272residualblock_3151274residualblock_3151276residualblock_3151278*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_residualblock_layer_call_and_return_conditional_losses_3150731?
'residualblock_1/StatefulPartitionedCallStatefulPartitionedCall.residualblock/StatefulPartitionedCall:output:0residualblock_1_3151281residualblock_1_3151283residualblock_1_3151285residualblock_1_3151287residualblock_1_3151289residualblock_1_3151291*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770?
'residualblock_2/StatefulPartitionedCallStatefulPartitionedCall0residualblock_1/StatefulPartitionedCall:output:0residualblock_2_3151294residualblock_2_3151296residualblock_2_3151298residualblock_2_3151300residualblock_2_3151302residualblock_2_3151304*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3150809?
'residualblock_3/StatefulPartitionedCallStatefulPartitionedCall0residualblock_2/StatefulPartitionedCall:output:0residualblock_3_3151307residualblock_3_3151309residualblock_3_3151311residualblock_3_3151313residualblock_3_3151315residualblock_3_3151317*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848?
'residualblock_4/StatefulPartitionedCallStatefulPartitionedCall0residualblock_3/StatefulPartitionedCall:output:0residualblock_4_3151320residualblock_4_3151322residualblock_4_3151324residualblock_4_3151326residualblock_4_3151328residualblock_4_3151330*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887?
'residualblock_5/StatefulPartitionedCallStatefulPartitionedCall0residualblock_4/StatefulPartitionedCall:output:0residualblock_5_3151333residualblock_5_3151335residualblock_5_3151337residualblock_5_3151339residualblock_5_3151341residualblock_5_3151343*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall0residualblock_5/StatefulPartitionedCall:output:0dense_35_3151346dense_35_3151348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_3150951d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlice$rescaling_3/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlice$rescaling_3/PartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
MulMul)dense_35/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_35/StatefulPartitionedCall&^residualblock/StatefulPartitionedCall(^residualblock_1/StatefulPartitionedCall(^residualblock_2/StatefulPartitionedCall(^residualblock_3/StatefulPartitionedCall(^residualblock_4/StatefulPartitionedCall(^residualblock_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2N
%residualblock/StatefulPartitionedCall%residualblock/StatefulPartitionedCall2R
'residualblock_1/StatefulPartitionedCall'residualblock_1/StatefulPartitionedCall2R
'residualblock_2/StatefulPartitionedCall'residualblock_2/StatefulPartitionedCall2R
'residualblock_3/StatefulPartitionedCall'residualblock_3/StatefulPartitionedCall2R
'residualblock_4/StatefulPartitionedCall'residualblock_4/StatefulPartitionedCall2R
'residualblock_5/StatefulPartitionedCall'residualblock_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_residualblock_5_layer_call_fn_3152584

inputs
unknown:22
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:22
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3150926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
1__inference_residualblock_3_layer_call_fn_3152500

inputs
unknown:22
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:22
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3150848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
1__inference_residualblock_4_layer_call_fn_3152542

inputs
unknown:22
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:22
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
1__inference_residualblock_1_layer_call_fn_3152416

inputs
unknown:22
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:22
	unknown_4:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3152609

inputs9
'dense_32_matmul_readvariableop_resource:226
(dense_32_biasadd_readvariableop_resource:29
'dense_33_matmul_readvariableop_resource:226
(dense_33_biasadd_readvariableop_resource:29
'dense_34_matmul_readvariableop_resource:226
(dense_34_biasadd_readvariableop_resource:2
identity??dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_32/MatMulMatMulinputs&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_32/TanhTanhdense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_33/MatMulMatMuldense_32/Tanh:y:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_34/MatMulMatMulinputs&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_5/addAddV2dense_33/BiasAdd:output:0dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_5/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?$
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152323

inputsG
5residualblock_dense_17_matmul_readvariableop_resource:2D
6residualblock_dense_17_biasadd_readvariableop_resource:2G
5residualblock_dense_18_matmul_readvariableop_resource:22D
6residualblock_dense_18_biasadd_readvariableop_resource:2G
5residualblock_dense_19_matmul_readvariableop_resource:2D
6residualblock_dense_19_biasadd_readvariableop_resource:2I
7residualblock_1_dense_20_matmul_readvariableop_resource:22F
8residualblock_1_dense_20_biasadd_readvariableop_resource:2I
7residualblock_1_dense_21_matmul_readvariableop_resource:22F
8residualblock_1_dense_21_biasadd_readvariableop_resource:2I
7residualblock_1_dense_22_matmul_readvariableop_resource:22F
8residualblock_1_dense_22_biasadd_readvariableop_resource:2I
7residualblock_2_dense_23_matmul_readvariableop_resource:22F
8residualblock_2_dense_23_biasadd_readvariableop_resource:2I
7residualblock_2_dense_24_matmul_readvariableop_resource:22F
8residualblock_2_dense_24_biasadd_readvariableop_resource:2I
7residualblock_2_dense_25_matmul_readvariableop_resource:22F
8residualblock_2_dense_25_biasadd_readvariableop_resource:2I
7residualblock_3_dense_26_matmul_readvariableop_resource:22F
8residualblock_3_dense_26_biasadd_readvariableop_resource:2I
7residualblock_3_dense_27_matmul_readvariableop_resource:22F
8residualblock_3_dense_27_biasadd_readvariableop_resource:2I
7residualblock_3_dense_28_matmul_readvariableop_resource:22F
8residualblock_3_dense_28_biasadd_readvariableop_resource:2I
7residualblock_4_dense_29_matmul_readvariableop_resource:22F
8residualblock_4_dense_29_biasadd_readvariableop_resource:2I
7residualblock_4_dense_30_matmul_readvariableop_resource:22F
8residualblock_4_dense_30_biasadd_readvariableop_resource:2I
7residualblock_4_dense_31_matmul_readvariableop_resource:22F
8residualblock_4_dense_31_biasadd_readvariableop_resource:2I
7residualblock_5_dense_32_matmul_readvariableop_resource:22F
8residualblock_5_dense_32_biasadd_readvariableop_resource:2I
7residualblock_5_dense_33_matmul_readvariableop_resource:22F
8residualblock_5_dense_33_biasadd_readvariableop_resource:2I
7residualblock_5_dense_34_matmul_readvariableop_resource:22F
8residualblock_5_dense_34_biasadd_readvariableop_resource:29
'dense_35_matmul_readvariableop_resource:26
(dense_35_biasadd_readvariableop_resource:
identity??dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?-residualblock/dense_17/BiasAdd/ReadVariableOp?,residualblock/dense_17/MatMul/ReadVariableOp?-residualblock/dense_18/BiasAdd/ReadVariableOp?,residualblock/dense_18/MatMul/ReadVariableOp?-residualblock/dense_19/BiasAdd/ReadVariableOp?,residualblock/dense_19/MatMul/ReadVariableOp?/residualblock_1/dense_20/BiasAdd/ReadVariableOp?.residualblock_1/dense_20/MatMul/ReadVariableOp?/residualblock_1/dense_21/BiasAdd/ReadVariableOp?.residualblock_1/dense_21/MatMul/ReadVariableOp?/residualblock_1/dense_22/BiasAdd/ReadVariableOp?.residualblock_1/dense_22/MatMul/ReadVariableOp?/residualblock_2/dense_23/BiasAdd/ReadVariableOp?.residualblock_2/dense_23/MatMul/ReadVariableOp?/residualblock_2/dense_24/BiasAdd/ReadVariableOp?.residualblock_2/dense_24/MatMul/ReadVariableOp?/residualblock_2/dense_25/BiasAdd/ReadVariableOp?.residualblock_2/dense_25/MatMul/ReadVariableOp?/residualblock_3/dense_26/BiasAdd/ReadVariableOp?.residualblock_3/dense_26/MatMul/ReadVariableOp?/residualblock_3/dense_27/BiasAdd/ReadVariableOp?.residualblock_3/dense_27/MatMul/ReadVariableOp?/residualblock_3/dense_28/BiasAdd/ReadVariableOp?.residualblock_3/dense_28/MatMul/ReadVariableOp?/residualblock_4/dense_29/BiasAdd/ReadVariableOp?.residualblock_4/dense_29/MatMul/ReadVariableOp?/residualblock_4/dense_30/BiasAdd/ReadVariableOp?.residualblock_4/dense_30/MatMul/ReadVariableOp?/residualblock_4/dense_31/BiasAdd/ReadVariableOp?.residualblock_4/dense_31/MatMul/ReadVariableOp?/residualblock_5/dense_32/BiasAdd/ReadVariableOp?.residualblock_5/dense_32/MatMul/ReadVariableOp?/residualblock_5/dense_33/BiasAdd/ReadVariableOp?.residualblock_5/dense_33/MatMul/ReadVariableOp?/residualblock_5/dense_34/BiasAdd/ReadVariableOp?.residualblock_5/dense_34/MatMul/ReadVariableOp[
rescaling_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2????Mb0?e
rescaling_3/CastCastrescaling_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
rescaling_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    f
rescaling_3/mulMulinputsrescaling_3/Cast:y:0*
T0*'
_output_shapes
:?????????~
rescaling_3/addAddV2rescaling_3/mul:z:0rescaling_3/Cast_1/x:output:0*
T0*'
_output_shapes
:??????????
,residualblock/dense_17/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_17_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
residualblock/dense_17/MatMulMatMulrescaling_3/add:z:04residualblock/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_17/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_17_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_17/BiasAddBiasAdd'residualblock/dense_17/MatMul:product:05residualblock/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2~
residualblock/dense_17/TanhTanh'residualblock/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
,residualblock/dense_18/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_18_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock/dense_18/MatMulMatMulresidualblock/dense_17/Tanh:y:04residualblock/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_18/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_18_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_18/BiasAddBiasAdd'residualblock/dense_18/MatMul:product:05residualblock/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
,residualblock/dense_19/MatMul/ReadVariableOpReadVariableOp5residualblock_dense_19_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
residualblock/dense_19/MatMulMatMulrescaling_3/add:z:04residualblock/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
-residualblock/dense_19/BiasAdd/ReadVariableOpReadVariableOp6residualblock_dense_19_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
residualblock/dense_19/BiasAddBiasAdd'residualblock/dense_19/MatMul:product:05residualblock/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock/add/addAddV2'residualblock/dense_18/BiasAdd:output:0'residualblock/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2g
residualblock/TanhTanhresidualblock/add/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_20/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_20_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_20/MatMulMatMulresidualblock/Tanh:y:06residualblock_1/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_20/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_20_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_20/BiasAddBiasAdd)residualblock_1/dense_20/MatMul:product:07residualblock_1/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_1/dense_20/TanhTanh)residualblock_1/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_21/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_21_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_21/MatMulMatMul!residualblock_1/dense_20/Tanh:y:06residualblock_1/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_21/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_21/BiasAddBiasAdd)residualblock_1/dense_21/MatMul:product:07residualblock_1/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_1/dense_22/MatMul/ReadVariableOpReadVariableOp7residualblock_1_dense_22_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_1/dense_22/MatMulMatMulresidualblock/Tanh:y:06residualblock_1/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_1/dense_22/BiasAdd/ReadVariableOpReadVariableOp8residualblock_1_dense_22_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_1/dense_22/BiasAddBiasAdd)residualblock_1/dense_22/MatMul:product:07residualblock_1/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_1/add_1/addAddV2)residualblock_1/dense_21/BiasAdd:output:0)residualblock_1/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_1/TanhTanhresidualblock_1/add_1/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_23/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_23_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_23/MatMulMatMulresidualblock_1/Tanh:y:06residualblock_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_23/BiasAddBiasAdd)residualblock_2/dense_23/MatMul:product:07residualblock_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_2/dense_23/TanhTanh)residualblock_2/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_24/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_24_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_24/MatMulMatMul!residualblock_2/dense_23/Tanh:y:06residualblock_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_24_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_24/BiasAddBiasAdd)residualblock_2/dense_24/MatMul:product:07residualblock_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_2/dense_25/MatMul/ReadVariableOpReadVariableOp7residualblock_2_dense_25_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_2/dense_25/MatMulMatMulresidualblock_1/Tanh:y:06residualblock_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp8residualblock_2_dense_25_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_2/dense_25/BiasAddBiasAdd)residualblock_2/dense_25/MatMul:product:07residualblock_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_2/add_2/addAddV2)residualblock_2/dense_24/BiasAdd:output:0)residualblock_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_2/TanhTanhresidualblock_2/add_2/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_26/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_26/MatMulMatMulresidualblock_2/Tanh:y:06residualblock_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_26/BiasAddBiasAdd)residualblock_3/dense_26/MatMul:product:07residualblock_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_3/dense_26/TanhTanh)residualblock_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_27/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_27/MatMulMatMul!residualblock_3/dense_26/Tanh:y:06residualblock_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_27/BiasAddBiasAdd)residualblock_3/dense_27/MatMul:product:07residualblock_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_3/dense_28/MatMul/ReadVariableOpReadVariableOp7residualblock_3_dense_28_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_3/dense_28/MatMulMatMulresidualblock_2/Tanh:y:06residualblock_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_3/dense_28/BiasAdd/ReadVariableOpReadVariableOp8residualblock_3_dense_28_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_3/dense_28/BiasAddBiasAdd)residualblock_3/dense_28/MatMul:product:07residualblock_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_3/add_3/addAddV2)residualblock_3/dense_27/BiasAdd:output:0)residualblock_3/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_3/TanhTanhresidualblock_3/add_3/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_29/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_29_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_29/MatMulMatMulresidualblock_3/Tanh:y:06residualblock_4/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_29/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_29_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_29/BiasAddBiasAdd)residualblock_4/dense_29/MatMul:product:07residualblock_4/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_4/dense_29/TanhTanh)residualblock_4/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_30/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_30_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_30/MatMulMatMul!residualblock_4/dense_29/Tanh:y:06residualblock_4/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_30/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_30/BiasAddBiasAdd)residualblock_4/dense_30/MatMul:product:07residualblock_4/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_4/dense_31/MatMul/ReadVariableOpReadVariableOp7residualblock_4_dense_31_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_4/dense_31/MatMulMatMulresidualblock_3/Tanh:y:06residualblock_4/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_4/dense_31/BiasAdd/ReadVariableOpReadVariableOp8residualblock_4_dense_31_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_4/dense_31/BiasAddBiasAdd)residualblock_4/dense_31/MatMul:product:07residualblock_4/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_4/add_4/addAddV2)residualblock_4/dense_30/BiasAdd:output:0)residualblock_4/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_4/TanhTanhresidualblock_4/add_4/add:z:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_32/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_32_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_32/MatMulMatMulresidualblock_4/Tanh:y:06residualblock_5/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_32/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_32_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_32/BiasAddBiasAdd)residualblock_5/dense_32/MatMul:product:07residualblock_5/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_5/dense_32/TanhTanh)residualblock_5/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_33/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_33_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_33/MatMulMatMul!residualblock_5/dense_32/Tanh:y:06residualblock_5/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_33/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_33_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_33/BiasAddBiasAdd)residualblock_5/dense_33/MatMul:product:07residualblock_5/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
.residualblock_5/dense_34/MatMul/ReadVariableOpReadVariableOp7residualblock_5_dense_34_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
residualblock_5/dense_34/MatMulMatMulresidualblock_4/Tanh:y:06residualblock_5/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
/residualblock_5/dense_34/BiasAdd/ReadVariableOpReadVariableOp8residualblock_5_dense_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
 residualblock_5/dense_34/BiasAddBiasAdd)residualblock_5/dense_34/MatMul:product:07residualblock_5/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
residualblock_5/add_5/addAddV2)residualblock_5/dense_33/BiasAdd:output:0)residualblock_5/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2m
residualblock_5/TanhTanhresidualblock_5/add_5/add:z:0*
T0*'
_output_shapes
:?????????2?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0?
dense_35/MatMulMatMulresidualblock_5/Tanh:y:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_35/ExpExpdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicerescaling_3/add:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSlicerescaling_3/add:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskj
SubSubstrided_slice:output:0strided_slice_1:output:0*
T0*#
_output_shapes
:?????????G
SquareSquareSub:z:0*
T0*#
_output_shapes
:?????????f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerescaling_3/add:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskZ
Square_1Squarestrided_slice_2:output:0*
T0*#
_output_shapes
:?????????T
AddAddV2
Square:y:0Square_1:y:0*
T0*#
_output_shapes
:?????????C
SqrtSqrtAdd:z:0*
T0*#
_output_shapes
:?????????^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   f
ReshapeReshapeSqrt:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????`
MulMuldense_35/Exp:y:0Reshape:output:0*
T0*'
_output_shapes
:?????????V
IdentityIdentityMul:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp.^residualblock/dense_17/BiasAdd/ReadVariableOp-^residualblock/dense_17/MatMul/ReadVariableOp.^residualblock/dense_18/BiasAdd/ReadVariableOp-^residualblock/dense_18/MatMul/ReadVariableOp.^residualblock/dense_19/BiasAdd/ReadVariableOp-^residualblock/dense_19/MatMul/ReadVariableOp0^residualblock_1/dense_20/BiasAdd/ReadVariableOp/^residualblock_1/dense_20/MatMul/ReadVariableOp0^residualblock_1/dense_21/BiasAdd/ReadVariableOp/^residualblock_1/dense_21/MatMul/ReadVariableOp0^residualblock_1/dense_22/BiasAdd/ReadVariableOp/^residualblock_1/dense_22/MatMul/ReadVariableOp0^residualblock_2/dense_23/BiasAdd/ReadVariableOp/^residualblock_2/dense_23/MatMul/ReadVariableOp0^residualblock_2/dense_24/BiasAdd/ReadVariableOp/^residualblock_2/dense_24/MatMul/ReadVariableOp0^residualblock_2/dense_25/BiasAdd/ReadVariableOp/^residualblock_2/dense_25/MatMul/ReadVariableOp0^residualblock_3/dense_26/BiasAdd/ReadVariableOp/^residualblock_3/dense_26/MatMul/ReadVariableOp0^residualblock_3/dense_27/BiasAdd/ReadVariableOp/^residualblock_3/dense_27/MatMul/ReadVariableOp0^residualblock_3/dense_28/BiasAdd/ReadVariableOp/^residualblock_3/dense_28/MatMul/ReadVariableOp0^residualblock_4/dense_29/BiasAdd/ReadVariableOp/^residualblock_4/dense_29/MatMul/ReadVariableOp0^residualblock_4/dense_30/BiasAdd/ReadVariableOp/^residualblock_4/dense_30/MatMul/ReadVariableOp0^residualblock_4/dense_31/BiasAdd/ReadVariableOp/^residualblock_4/dense_31/MatMul/ReadVariableOp0^residualblock_5/dense_32/BiasAdd/ReadVariableOp/^residualblock_5/dense_32/MatMul/ReadVariableOp0^residualblock_5/dense_33/BiasAdd/ReadVariableOp/^residualblock_5/dense_33/MatMul/ReadVariableOp0^residualblock_5/dense_34/BiasAdd/ReadVariableOp/^residualblock_5/dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2^
-residualblock/dense_17/BiasAdd/ReadVariableOp-residualblock/dense_17/BiasAdd/ReadVariableOp2\
,residualblock/dense_17/MatMul/ReadVariableOp,residualblock/dense_17/MatMul/ReadVariableOp2^
-residualblock/dense_18/BiasAdd/ReadVariableOp-residualblock/dense_18/BiasAdd/ReadVariableOp2\
,residualblock/dense_18/MatMul/ReadVariableOp,residualblock/dense_18/MatMul/ReadVariableOp2^
-residualblock/dense_19/BiasAdd/ReadVariableOp-residualblock/dense_19/BiasAdd/ReadVariableOp2\
,residualblock/dense_19/MatMul/ReadVariableOp,residualblock/dense_19/MatMul/ReadVariableOp2b
/residualblock_1/dense_20/BiasAdd/ReadVariableOp/residualblock_1/dense_20/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_20/MatMul/ReadVariableOp.residualblock_1/dense_20/MatMul/ReadVariableOp2b
/residualblock_1/dense_21/BiasAdd/ReadVariableOp/residualblock_1/dense_21/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_21/MatMul/ReadVariableOp.residualblock_1/dense_21/MatMul/ReadVariableOp2b
/residualblock_1/dense_22/BiasAdd/ReadVariableOp/residualblock_1/dense_22/BiasAdd/ReadVariableOp2`
.residualblock_1/dense_22/MatMul/ReadVariableOp.residualblock_1/dense_22/MatMul/ReadVariableOp2b
/residualblock_2/dense_23/BiasAdd/ReadVariableOp/residualblock_2/dense_23/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_23/MatMul/ReadVariableOp.residualblock_2/dense_23/MatMul/ReadVariableOp2b
/residualblock_2/dense_24/BiasAdd/ReadVariableOp/residualblock_2/dense_24/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_24/MatMul/ReadVariableOp.residualblock_2/dense_24/MatMul/ReadVariableOp2b
/residualblock_2/dense_25/BiasAdd/ReadVariableOp/residualblock_2/dense_25/BiasAdd/ReadVariableOp2`
.residualblock_2/dense_25/MatMul/ReadVariableOp.residualblock_2/dense_25/MatMul/ReadVariableOp2b
/residualblock_3/dense_26/BiasAdd/ReadVariableOp/residualblock_3/dense_26/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_26/MatMul/ReadVariableOp.residualblock_3/dense_26/MatMul/ReadVariableOp2b
/residualblock_3/dense_27/BiasAdd/ReadVariableOp/residualblock_3/dense_27/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_27/MatMul/ReadVariableOp.residualblock_3/dense_27/MatMul/ReadVariableOp2b
/residualblock_3/dense_28/BiasAdd/ReadVariableOp/residualblock_3/dense_28/BiasAdd/ReadVariableOp2`
.residualblock_3/dense_28/MatMul/ReadVariableOp.residualblock_3/dense_28/MatMul/ReadVariableOp2b
/residualblock_4/dense_29/BiasAdd/ReadVariableOp/residualblock_4/dense_29/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_29/MatMul/ReadVariableOp.residualblock_4/dense_29/MatMul/ReadVariableOp2b
/residualblock_4/dense_30/BiasAdd/ReadVariableOp/residualblock_4/dense_30/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_30/MatMul/ReadVariableOp.residualblock_4/dense_30/MatMul/ReadVariableOp2b
/residualblock_4/dense_31/BiasAdd/ReadVariableOp/residualblock_4/dense_31/BiasAdd/ReadVariableOp2`
.residualblock_4/dense_31/MatMul/ReadVariableOp.residualblock_4/dense_31/MatMul/ReadVariableOp2b
/residualblock_5/dense_32/BiasAdd/ReadVariableOp/residualblock_5/dense_32/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_32/MatMul/ReadVariableOp.residualblock_5/dense_32/MatMul/ReadVariableOp2b
/residualblock_5/dense_33/BiasAdd/ReadVariableOp/residualblock_5/dense_33/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_33/MatMul/ReadVariableOp.residualblock_5/dense_33/MatMul/ReadVariableOp2b
/residualblock_5/dense_34/BiasAdd/ReadVariableOp/residualblock_5/dense_34/BiasAdd/ReadVariableOp2`
.residualblock_5/dense_34/MatMul/ReadVariableOp.residualblock_5/dense_34/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3150770

inputs9
'dense_20_matmul_readvariableop_resource:226
(dense_20_biasadd_readvariableop_resource:29
'dense_21_matmul_readvariableop_resource:226
(dense_21_biasadd_readvariableop_resource:29
'dense_22_matmul_readvariableop_resource:226
(dense_22_biasadd_readvariableop_resource:2
identity??dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_21/MatMulMatMuldense_20/Tanh:y:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_22/MatMulMatMulinputs&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_1/addAddV2dense_21/BiasAdd:output:0dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_1/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?	
3__inference_travel_times_nn_3_layer_call_fn_3151532
input_1
unknown:2
	unknown_0:2
	unknown_1:22
	unknown_2:2
	unknown_3:2
	unknown_4:2
	unknown_5:22
	unknown_6:2
	unknown_7:22
	unknown_8:2
	unknown_9:22

unknown_10:2

unknown_11:22

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:22

unknown_16:2

unknown_17:22

unknown_18:2

unknown_19:22

unknown_20:2

unknown_21:22

unknown_22:2

unknown_23:22

unknown_24:2

unknown_25:22

unknown_26:2

unknown_27:22

unknown_28:2

unknown_29:22

unknown_30:2

unknown_31:22

unknown_32:2

unknown_33:22

unknown_34:2

unknown_35:2

unknown_36:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3150887

inputs9
'dense_29_matmul_readvariableop_resource:226
(dense_29_biasadd_readvariableop_resource:29
'dense_30_matmul_readvariableop_resource:226
(dense_30_biasadd_readvariableop_resource:29
'dense_31_matmul_readvariableop_resource:226
(dense_31_biasadd_readvariableop_resource:2
identity??dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_29/MatMulMatMulinputs&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2b
dense_29/TanhTanhdense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0?
dense_30/MatMulMatMuldense_29/Tanh:y:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:22*
dtype0{
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2z
	add_4/addAddV2dense_30/BiasAdd:output:0dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2M
TanhTanhadd_4/add:z:0*
T0*'
_output_shapes
:?????????2W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????2?
NoOpNoOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2: : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
input_layer
	rescaling

denses
output_layer
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	decay
learning_ratem?m? m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?"
	optimizer
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
420
521
622
723
824
925
:26
;27
<28
=29
>30
?31
@32
A33
B34
C35
36
37"
trackable_list_wrapper
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
420
521
622
723
824
925
:26
;27
<28
=29
>30
?31
@32
A33
B34
C35
36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Ndense_1
Odense_2
Pdense_3
Qadd
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Vdense_1
Wdense_2
Xdense_3
Yadd
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^dense_1
_dense_2
`dense_3
aadd
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
fdense_1
gdense_2
hdense_3
iadd
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
ndense_1
odense_2
pdense_3
qadd
r	variables
strainable_variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
vdense_1
wdense_2
xdense_3
yadd
z	variables
{trainable_variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
3:122!travel_times_nn_3/dense_35/kernel
-:+2travel_times_nn_3/dense_35/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
A:?22/travel_times_nn_3/residualblock/dense_17/kernel
;:922-travel_times_nn_3/residualblock/dense_17/bias
A:?222/travel_times_nn_3/residualblock/dense_18/kernel
;:922-travel_times_nn_3/residualblock/dense_18/bias
A:?22/travel_times_nn_3/residualblock/dense_19/kernel
;:922-travel_times_nn_3/residualblock/dense_19/bias
C:A2221travel_times_nn_3/residualblock_1/dense_20/kernel
=:;22/travel_times_nn_3/residualblock_1/dense_20/bias
C:A2221travel_times_nn_3/residualblock_1/dense_21/kernel
=:;22/travel_times_nn_3/residualblock_1/dense_21/bias
C:A2221travel_times_nn_3/residualblock_1/dense_22/kernel
=:;22/travel_times_nn_3/residualblock_1/dense_22/bias
C:A2221travel_times_nn_3/residualblock_2/dense_23/kernel
=:;22/travel_times_nn_3/residualblock_2/dense_23/bias
C:A2221travel_times_nn_3/residualblock_2/dense_24/kernel
=:;22/travel_times_nn_3/residualblock_2/dense_24/bias
C:A2221travel_times_nn_3/residualblock_2/dense_25/kernel
=:;22/travel_times_nn_3/residualblock_2/dense_25/bias
C:A2221travel_times_nn_3/residualblock_3/dense_26/kernel
=:;22/travel_times_nn_3/residualblock_3/dense_26/bias
C:A2221travel_times_nn_3/residualblock_3/dense_27/kernel
=:;22/travel_times_nn_3/residualblock_3/dense_27/bias
C:A2221travel_times_nn_3/residualblock_3/dense_28/kernel
=:;22/travel_times_nn_3/residualblock_3/dense_28/bias
C:A2221travel_times_nn_3/residualblock_4/dense_29/kernel
=:;22/travel_times_nn_3/residualblock_4/dense_29/bias
C:A2221travel_times_nn_3/residualblock_4/dense_30/kernel
=:;22/travel_times_nn_3/residualblock_4/dense_30/bias
C:A2221travel_times_nn_3/residualblock_4/dense_31/kernel
=:;22/travel_times_nn_3/residualblock_4/dense_31/bias
C:A2221travel_times_nn_3/residualblock_5/dense_32/kernel
=:;22/travel_times_nn_3/residualblock_5/dense_32/bias
C:A2221travel_times_nn_3/residualblock_5/dense_33/kernel
=:;22/travel_times_nn_3/residualblock_5/dense_33/bias
C:A2221travel_times_nn_3/residualblock_5/dense_34/kernel
=:;22/travel_times_nn_3/residualblock_5/dense_34/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

 kernel
!bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
 0
!1
"2
#3
$4
%5"
trackable_list_wrapper
J
 0
!1
"2
#3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

&kernel
'bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

,kernel
-bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

2kernel
3bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
5bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
20
31
42
53
64
75"
trackable_list_wrapper
J
20
31
42
53
64
75"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

8kernel
9bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
80
91
:2
;3
<4
=5"
trackable_list_wrapper
J
80
91
:2
;3
<4
=5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
r	variables
strainable_variables
tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
>0
?1
@2
A3
B4
C5"
trackable_list_wrapper
J
>0
?1
@2
A3
B4
C5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
n0
o1
p2
q3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8:622(Adam/travel_times_nn_3/dense_35/kernel/m
2:02&Adam/travel_times_nn_3/dense_35/bias/m
F:D226Adam/travel_times_nn_3/residualblock/dense_17/kernel/m
@:>224Adam/travel_times_nn_3/residualblock/dense_17/bias/m
F:D2226Adam/travel_times_nn_3/residualblock/dense_18/kernel/m
@:>224Adam/travel_times_nn_3/residualblock/dense_18/bias/m
F:D226Adam/travel_times_nn_3/residualblock/dense_19/kernel/m
@:>224Adam/travel_times_nn_3/residualblock/dense_19/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_1/dense_20/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_1/dense_21/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_1/dense_22/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_2/dense_23/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_2/dense_24/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_2/dense_25/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_3/dense_26/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_3/dense_27/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_3/dense_28/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_4/dense_29/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_4/dense_30/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_4/dense_31/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_5/dense_32/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_5/dense_33/bias/m
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/m
B:@226Adam/travel_times_nn_3/residualblock_5/dense_34/bias/m
8:622(Adam/travel_times_nn_3/dense_35/kernel/v
2:02&Adam/travel_times_nn_3/dense_35/bias/v
F:D226Adam/travel_times_nn_3/residualblock/dense_17/kernel/v
@:>224Adam/travel_times_nn_3/residualblock/dense_17/bias/v
F:D2226Adam/travel_times_nn_3/residualblock/dense_18/kernel/v
@:>224Adam/travel_times_nn_3/residualblock/dense_18/bias/v
F:D226Adam/travel_times_nn_3/residualblock/dense_19/kernel/v
@:>224Adam/travel_times_nn_3/residualblock/dense_19/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_20/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_1/dense_20/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_21/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_1/dense_21/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_1/dense_22/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_1/dense_22/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_23/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_2/dense_23/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_24/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_2/dense_24/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_2/dense_25/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_2/dense_25/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_26/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_3/dense_26/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_27/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_3/dense_27/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_3/dense_28/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_3/dense_28/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_29/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_4/dense_29/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_30/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_4/dense_30/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_4/dense_31/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_4/dense_31/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_32/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_5/dense_32/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_33/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_5/dense_33/bias/v
H:F2228Adam/travel_times_nn_3/residualblock_5/dense_34/kernel/v
B:@226Adam/travel_times_nn_3/residualblock_5/dense_34/bias/v
?2?
3__inference_travel_times_nn_3_layer_call_fn_3151057
3__inference_travel_times_nn_3_layer_call_fn_3151918
3__inference_travel_times_nn_3_layer_call_fn_3151999
3__inference_travel_times_nn_3_layer_call_fn_3151532?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152161
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152323
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151640
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151748?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference__wrapped_model_3150688input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_rescaling_3_layer_call_fn_3152328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3152337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_35_layer_call_fn_3152346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_35_layer_call_and_return_conditional_losses_3152357?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3151837input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_residualblock_layer_call_fn_3152374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_residualblock_layer_call_and_return_conditional_losses_3152399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_residualblock_1_layer_call_fn_3152416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3152441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_residualblock_2_layer_call_fn_3152458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3152483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_residualblock_3_layer_call_fn_3152500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3152525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_residualblock_4_layer_call_fn_3152542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3152567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_residualblock_5_layer_call_fn_3152584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3152609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_3150688?& !"#$%&'()*+,-./0123456789:;<=>?@ABC0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
E__inference_dense_35_layer_call_and_return_conditional_losses_3152357\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? }
*__inference_dense_35_layer_call_fn_3152346O/?,
%?"
 ?
inputs?????????2
? "???????????
H__inference_rescaling_3_layer_call_and_return_conditional_losses_3152337X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_rescaling_3_layer_call_fn_3152328K/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_residualblock_1_layer_call_and_return_conditional_losses_3152441`&'()*+/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? ?
1__inference_residualblock_1_layer_call_fn_3152416S&'()*+/?,
%?"
 ?
inputs?????????2
? "??????????2?
L__inference_residualblock_2_layer_call_and_return_conditional_losses_3152483`,-./01/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? ?
1__inference_residualblock_2_layer_call_fn_3152458S,-./01/?,
%?"
 ?
inputs?????????2
? "??????????2?
L__inference_residualblock_3_layer_call_and_return_conditional_losses_3152525`234567/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? ?
1__inference_residualblock_3_layer_call_fn_3152500S234567/?,
%?"
 ?
inputs?????????2
? "??????????2?
L__inference_residualblock_4_layer_call_and_return_conditional_losses_3152567`89:;<=/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? ?
1__inference_residualblock_4_layer_call_fn_3152542S89:;<=/?,
%?"
 ?
inputs?????????2
? "??????????2?
L__inference_residualblock_5_layer_call_and_return_conditional_losses_3152609`>?@ABC/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? ?
1__inference_residualblock_5_layer_call_fn_3152584S>?@ABC/?,
%?"
 ?
inputs?????????2
? "??????????2?
J__inference_residualblock_layer_call_and_return_conditional_losses_3152399` !"#$%/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????2
? ?
/__inference_residualblock_layer_call_fn_3152374S !"#$%/?,
%?"
 ?
inputs?????????
? "??????????2?
%__inference_signature_wrapper_3151837?& !"#$%&'()*+,-./0123456789:;<=>?@ABC;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1??????????
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151640?& !"#$%&'()*+,-./0123456789:;<=>?@ABC4?1
*?'
!?
input_1?????????
p 
? "%?"
?
0?????????
? ?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3151748?& !"#$%&'()*+,-./0123456789:;<=>?@ABC4?1
*?'
!?
input_1?????????
p
? "%?"
?
0?????????
? ?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152161?& !"#$%&'()*+,-./0123456789:;<=>?@ABC3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
N__inference_travel_times_nn_3_layer_call_and_return_conditional_losses_3152323?& !"#$%&'()*+,-./0123456789:;<=>?@ABC3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
3__inference_travel_times_nn_3_layer_call_fn_3151057x& !"#$%&'()*+,-./0123456789:;<=>?@ABC4?1
*?'
!?
input_1?????????
p 
? "???????????
3__inference_travel_times_nn_3_layer_call_fn_3151532x& !"#$%&'()*+,-./0123456789:;<=>?@ABC4?1
*?'
!?
input_1?????????
p
? "???????????
3__inference_travel_times_nn_3_layer_call_fn_3151918w& !"#$%&'()*+,-./0123456789:;<=>?@ABC3?0
)?&
 ?
inputs?????????
p 
? "???????????
3__inference_travel_times_nn_3_layer_call_fn_3151999w& !"#$%&'()*+,-./0123456789:;<=>?@ABC3?0
)?&
 ?
inputs?????????
p
? "??????????