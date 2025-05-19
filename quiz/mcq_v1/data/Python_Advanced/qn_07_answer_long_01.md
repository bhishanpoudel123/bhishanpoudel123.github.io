Sure! Here's a simple example of using a `@staticmethod` in a Python class:

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

# Usage
result = MathUtils.add(3, 5)
print("The result is:", result)
```

### Key points:

* `@staticmethod` is used when a method doesn't need access to the instance (`self`) or class (`cls`) — it just belongs to the class's namespace.
* You can call it on the class itself without creating an instance.

Let me know if you want an example with `@classmethod` for comparison.

Great! Here's an example that shows both `@staticmethod` and `@classmethod` in the same class, so you can compare their usage:

```python
class Circle:
    pi = 3.14159

    def __init__(self, radius):
        self.radius = radius

    @staticmethod
    def area(radius):
        return Circle.pi * radius * radius

    @classmethod
    def from_diameter(cls, diameter):
        radius = diameter / 2
        return cls(radius)

# Usage of staticmethod
print("Area with radius 5:", Circle.area(5))

# Usage of classmethod
circle = Circle.from_diameter(10)
print("Radius from diameter 10:", circle.radius)
```

### Explanation:

* **`@staticmethod area()`**: Just takes a radius and returns the area. It doesn’t care about the class or instance.
* **`@classmethod from_diameter()`**: Knows about the class (`cls`) and returns an instance using alternative input (diameter).
