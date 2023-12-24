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


### Tensorflow regression

### Tensorflow classification

### Tensorflow computer vision

### Transfer learning Part 1: Feature extraction

### Transfer learning Part 2: Fine-turning

### Transfer learning Part 3: Scaling up

### Tensorflow NLP Fundamentals

### Tensorflow Time series Fundamentals

### Preparing to Pass the Tensorflow Developer Certification exam
