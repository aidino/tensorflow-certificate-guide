# tensorflow-certificate-guide
TensorFlow Developer Certificate Guide: Efficiently tackle deep learning and ML problems to ace the Developer Certificate exam

Resource:
- https://github.com/PacktPublishing/TensorFlow-Developer-Certificate-Guide
- https://dev.mrdbourke.com/tensorflow-deep-learning/

---

### Tensorflow fundamentals

#### Basic

- **scalar**:
```commandline
scalar = tf.constant(7)
```
- **vector**: 
```commandline
vector = tf.constant([10, 10])
```
- **matrix**: 
```python
matrix = tf.constant([[10, 7],
                      [7, 10]])
```
- **tensor (n-dim)**: 
```python
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
```
- **variable**
```python
# Create variable
variable = tf.Variable([1, 2]) # [1, 2]
# get value
variable[0] # 1
# assign value
variable[0].assign(2) # [2, 2]
```
- **Creating random tensor**
```python
generator = tf.random.Generator.from_seed(42)
random = generator.normal([2, 3]) # shape (2, 3)
```

- **Shuffle a tensor**
```python
tensor = tf.constant([[1, 2], [3, 4], [5, 6]])
# array([[1, 2],
#        [3, 4],
#        [5, 6]], dtype=int32)>
shuffle_tensor = tf.random.shuffle(tensor, seed=42)
# array([[5, 6],
#        [3, 4],
#        [1, 2]], dtype=int32)>
```
- **The other way to make tensor**
```python
# Tensor ones
tf.ones(shape=(2, 3))
# array([[1., 1., 1.],
#        [1., 1., 1.]], dtype=float32)>

# Tensor zeros
tf.zeros(shape=(2, 3))
# array([[0., 0., 0.],
#        [0., 0., 0.]], dtype=float32)>

```
- **Getting infomation from tensor (shape, rank, size)**
```python
rank_4_tensor = tf.zeros([2, 3, 4, 5])
rank_4_tensor.shape # TensorShape([2, 3, 4, 5])
rank_4_tensor.ndim # 4 (rank)
tf.size(rank_4_tensor) # <tf.Tensor: shape=(), dtype=int32, numpy=120>
rank_4_tensor.dtype # tf.float32
# Elements along axis 0 of tensor
rank_4_tensor.shape[0] # 2
# Elements along last axis of tensor
rank_4_tensor.shape[-1] # 5
# Total number of elements (2*3*4*5)
tf.size(rank_4_tensor).numpy() # 120
# You can also index tensors just like Python lists.
rank_4_tensor[:2, :2, :2, :2]
# Get the last item of each row
rank_4_tensor[:, -1]
# Add an extra dimension (to the end)
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # in Python "..." means "all dimensions prior to"
# of
tf.expand_dims(rank_3_tensor, axis=-1)
```

#### Manupulating tensors (tensor operator)

- **Basic Operator**

```python
tensor = tf.constant([[1, 2], [3, 4]])
# array([[1, 2],
#        [3, 4]], dtype=int32)>

tensor + 10
# array([[11, 12],
#        [13, 14]], dtype=int32)>

tensor * 10
# array([[10, 20],
#        [30, 40]], dtype=int32)>

# Use the tensorflow function equivalent of the '*' (multiply) operator
tf.multiply(tensor, 10)
# array([[10, 20],
#        [30, 40]], dtype=int32)>


```

- **Matrix Multiplication**
```python
tf.matmul(tensor, tensor)
# array([[ 7, 10],
#        [15, 22]], dtype=int32)>
tensor @ tensor
# array([[ 7, 10],
#        [15, 22]], dtype=int32)>

```
- `tf.reshape(tensor, shape=(axb))`

Hàm reshape trong TensorFlow được sử dụng để thay đổi hình dạng của một tensor, nghĩa là bạn có thể điều chỉnh kích thước của tensor mà không làm thay đổi dữ liệu bên trong nó. 
```python
tensor = tf.constant([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])

tf.reshape(tensor, shape=(4, 3))
# <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]], dtype=int32)>

```
- `tf.transpose(tensor)`

Hàm transpose trong TensorFlow được sử dụng để hoán đổi chiều của tensor

```python
tf.transpose(tensor)

# <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
# array([[ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11],
#        [ 4,  8, 12]], dtype=int32)>

```

- `tf.matmul(tensorA, tensorB)` vs `tf.tensordot(tensorA, tensorB, axes)`

`tf.matmul`:

**Mục đích chính**: Dùng để thực hiện phép nhân ma trận giữa hai tensor.

**Cách sử dụng**: `tf.matmul(a, b)`, trong đó `a` và `b` là hai tensor có thể là ma trận hoặc đa chiều tensor.

**Phù hợp cho**: Các phép nhân ma trận truyền thống và các phép toán tuyến tính khác.

`tf.tensordot`:

**Mục đích chính**: Dùng để thực hiện các phép nhân tensor và giảm chiều của tensor đầu vào thông qua việc chỉ định các chiều cần được nhân.

**Cách sử dụng**: `tf.tensordot(a, b, axes)`, trong đó `a` và `b` là hai tensor cần nhân, và `axes` là một tuple chứa các cặp chiều tương ứng cần nhân.

**Phù hợp cho**: Các phép nhân tensor linh hoạt, giảm chiều tensor.

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
tf.tensordot(a, b, axes=1)
# array([[19, 22],
#        [43, 50]], dtype=int32)>

```

Tóm lại, `tf.matmul` được thiết kế chủ yếu cho các phép nhân ma trận, trong khi `tf.tensordot` có tính linh hoạt cao hơn, cho phép thực hiện các phép nhân tensor với việc tùy chọn giảm chiều. 
Cả hai đều có vai trò quan trọng trong xử lý tensor và thực hiện các phép toán tuyến tính trong TensorFlow.

- **Change the data type of a tensor**
```python 
# Define a tensor with data type
tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float16)

# Change the data type 
tensor = tf.cast(tensor, dtype=tf.float32)
```
- **Getting the absolute value**
```python
tensor = tf.constant([-1, -2])

tf.abs(tensor) # [1, 2]
```
- **min, max, mean, sum (aggregation)**

```python 
E = tf.constant(np.random.randint(low=0, high=100, size=20))
# array([24, 49, 79, 61, 20,  2, 75, 87, 34, 51, 81, 59,  8, 28, 25, 11, 83, 87, 45, 55])>
tf.reduce_min(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=2>
tf.reduce_max(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=87>
tf.reduce_mean(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=48>
 tf.reduce_sum(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=964>

# Finding the positional maximum and minimum
tf.argmax(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=7>
tf.argmin(E)
# <tf.Tensor: shape=(), dtype=int64, numpy=5>

```

- **Squeezing a tensor (removing all single dimensions)**

```python
G = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
# array([[[[[23, 73, 56, 71, 66, 97, 13, 61, 94, 37, 29, 81, 81, 74, 56,
#            40, 72, 21, 40, 60,  6, 85, 52, 86,  6, 61, 58, 70, 98,  1,
#            55,  1, 22, 70, 17, 42, 90, 63, 20, 59, 22, 49, 66, 33, 81,
#            84, 74, 85, 28, 72]]]]])>
G.shape # TensorShape([1, 1, 1, 1, 50])
G.ndim # 5

G_squeezed = tf.squeeze(G)
# array([23, 73, 56, 71, 66, 97, 13, 61, 94, 37, 29, 81, 81, 74, 56, 40, 72,
#        21, 40, 60,  6, 85, 52, 86,  6, 61, 58, 70, 98,  1, 55,  1, 22, 70,
#        17, 42, 90, 63, 20, 59, 22, 49, 66, 33, 81, 84, 74, 85, 28, 72])>

G_squeezed.shape # TensorShape([50])
G_squeezed.ndim # 1
```
### Tensorflow regression

### Tensorflow classification

### Tensorflow computer vision

### Transfer learning Part 1: Feature extraction

### Transfer learning Part 2: Fine-turning

### Transfer learning Part 3: Scaling up

### Tensorflow NLP Fundamentals

### Tensorflow Time series Fundamentals

### Preparing to Pass the Tensorflow Developer Certification exam
