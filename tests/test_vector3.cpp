#include "common/common_types.hpp"
#include "vector3/vector3.hpp"

#include "gtest/gtest.h"

namespace RayTracer
{

TEST(Vector3, Add)
{
    vector3f32 left{1.0f, 2.0f, 3.0f};
    vector3f32 right{4.0f, 5.0f, 6.0f};

    vector3f32 result = left + right;

    EXPECT_EQ(result.x(), left.x() + right.x());
    EXPECT_EQ(result.y(), left.y() + right.y());
    EXPECT_EQ(result.z(), left.z() + right.z());

    left += right;
    EXPECT_EQ(left.x(), result.x());
    EXPECT_EQ(left.y(), result.y());
    EXPECT_EQ(left.z(), result.z());
}

TEST(Vector3, Subtract)
{
    vector3f32 left{1.0f, 2.0f, 3.0f};
    vector3f32 right{4.0f, 5.0f, 6.0f};

    vector3f32 result = left - right;

    EXPECT_EQ(result.x(), left.x() - right.x());
    EXPECT_EQ(result.y(), left.y() - right.y());
    EXPECT_EQ(result.z(), left.z() - right.z());

    left -= right;
    EXPECT_EQ(left.x(), result.x());
    EXPECT_EQ(left.y(), result.y());
    EXPECT_EQ(left.z(), result.z());
}

TEST(Vector3, Multiply)
{
    constexpr f32 t{2.0f};
    vector3f32 left{1.0f, 2.0f, 3.0f};
    vector3f32 right{4.0f, 5.0f, 6.0f};

    vector3f32 result = left * right;

    EXPECT_EQ(result.x(), left.x() * right.x());
    EXPECT_EQ(result.y(), left.y() * right.y());
    EXPECT_EQ(result.z(), left.z() * right.z());

    result = t * left;
    EXPECT_EQ(result.x(), t * left.x());
    EXPECT_EQ(result.y(), t * left.y());
    EXPECT_EQ(result.z(), t * left.z());

    left *= t;
    EXPECT_EQ(left.x(), result.x());
    EXPECT_EQ(left.y(), result.y());
    EXPECT_EQ(left.z(), result.z());
}

TEST(Vector3, Divide)
{
    constexpr float t{2.0f};
    vector3f32 vec{2.0f, 4.0f, 6.0f};

    vector3f32 result = vec / t;

    EXPECT_EQ(result.x(), vec.x() / t);
    EXPECT_EQ(result.y(), vec.y() / t);
    EXPECT_EQ(result.z(), vec.z() / t);

    vec /= t;
    EXPECT_EQ(vec.x(), result.x());
    EXPECT_EQ(vec.y(), result.y());
    EXPECT_EQ(vec.z(), result.z());
}

TEST(Vector3, Magnitude)
{
    vector3f32 vec{1.0f, 4.0f, 8.0f};
    auto magnitude_sqr = vec.magnitude_squared();
    EXPECT_EQ(magnitude_sqr, 81);
    auto magnitude = vec.magnitude_host();
    EXPECT_EQ(magnitude, 9);
}

TEST(Vector3, Normalize)
{
    vector3f32 vec{1.0f, 2.0f, 3.0f};

    vector3f32 result = normalize_host(vec);
    EXPECT_FLOAT_EQ(result.x(), 0.2672612419124244);
    EXPECT_FLOAT_EQ(result.y(), 0.5345224838248488);
    EXPECT_FLOAT_EQ(result.z(), 0.8017837257372732);
    EXPECT_FLOAT_EQ(result.magnitude_host(), 1.0f);

    vec.normalize_host();
    EXPECT_FLOAT_EQ(vec.x(), result.x());
    EXPECT_FLOAT_EQ(vec.y(), result.y());
    EXPECT_FLOAT_EQ(vec.z(), result.z());
    EXPECT_FLOAT_EQ(vec.magnitude_host(), result.magnitude_host());
}

TEST(Vector3, DotAndCross)
{
    vector3f32 left{1.0f, 2.0f, 3.0f};
    vector3f32 right{4.0f, 5.0f, 6.0f};

    f32 dot_result = dot(left, right);

    EXPECT_FLOAT_EQ(dot_result, 32.0f);

    vector3f32 cross_result = cross(left, right);

    EXPECT_FLOAT_EQ(cross_result.x(), -3.0f);
    EXPECT_FLOAT_EQ(cross_result.y(), 6.0f);
    EXPECT_FLOAT_EQ(cross_result.z(), -3.0f);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
