//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/Constant.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
// FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//

// Number of base-10 digits in an unsigned integer
int num_digits(unsigned long val);

// Return a replacement string if the given string is too long
char const*
trunc_string(unsigned int digits, char const* str, char const* trunc);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get a "least common denominator" for soft comparisons.
 */
template<class T1, class T2>
struct SoftPrecisionType
{
    using type = void;
};

template<class T>
struct SoftPrecisionType<T, T>
{
    using type = T;
};

// When comparing doubles to floats, use the floating point epsilon for
// comparison
template<>
struct SoftPrecisionType<double, float>
{
    using type = float;
};

template<>
struct SoftPrecisionType<float, double>
{
    using type = float;
};

// Allow reference type to be an int (i.e. user writes 0 or 1 instead of 0.)
template<class T>
struct SoftPrecisionType<int, T>
{
    using type = T;
};

// Allow reference type to be a constant
template<class T>
struct SoftPrecisionType<Constant, T>
{
    using type = T;
};

// Allow actual type to be a constant (used in Constants test)
template<class T>
struct SoftPrecisionType<T, Constant>
{
    using type = T;
};

// Allow both to be constants
template<>
struct SoftPrecisionType<Constant, Constant>
{
    using type = double;
};

//---------------------------------------------------------------------------//
//! Whether soft equivalence can be performed on the given types.
template<class T1, class T2>
constexpr bool can_soft_equiv()
{
    return !std::is_same_v<typename SoftPrecisionType<T1, T2>::type, void>;
}

//---------------------------------------------------------------------------//
//! Compare a range of values.
template<class BinaryOp>
::testing::AssertionResult
IsSoftEquivImpl(typename BinaryOp::value_type expected,
                char const* expected_expr,
                typename BinaryOp::value_type actual,
                char const* actual_expr,
                BinaryOp comp)
{
    using value_type = typename BinaryOp::value_type;

    if (comp(expected, actual))
    {
        return ::testing::AssertionSuccess();
    }

    // Failed: print nice error message
    ::testing::AssertionResult result = ::testing::AssertionFailure();

    result << "Value of: " << actual_expr << "\n  Actual: " << actual
           << "\nExpected: " << expected_expr << "\nWhich is: " << expected
           << '\n';

    if (SoftZero<value_type>{comp.abs()}(expected))
    {
        // Avoid divide by zero errors
        result << "(Absolute error " << actual - expected
               << " exceeds tolerance " << comp.abs() << ")";
    }
    else
    {
        result << "(Relative error " << (actual - expected) / expected
               << " exceeds tolerance " << comp.rel() << ")";
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Predicate for relative error soft equivalence.
 */
template<class Value_E, class Value_A>
::testing::AssertionResult IsSoftEquiv(char const* expected_expr,
                                       char const* actual_expr,
                                       Value_E&& expected,
                                       Value_A&& actual)
{
    using VE = std::remove_cv_t<std::remove_reference_t<Value_E>>;
    using VA = std::remove_cv_t<std::remove_reference_t<Value_A>>;

    static_assert(can_soft_equiv<VE, VA>(),
                  "Invalid types for soft equivalence");

    // Construct with automatic or specified tolerances
    using ValueT = typename SoftPrecisionType<VE, VA>::type;
    using BinaryOp = EqualOr<SoftEqual<ValueT>>;

    return IsSoftEquivImpl(static_cast<ValueT>(expected),
                           expected_expr,
                           static_cast<ValueT>(actual),
                           actual_expr,
                           BinaryOp{});
}

//---------------------------------------------------------------------------//
/*!
 * Predicate for relative error soft equivalence.
 */
template<class Value_E, class Value_A, class Value_R>
::testing::AssertionResult IsSoftEquiv(char const* expected_expr,
                                       char const* actual_expr,
                                       char const*,
                                       Value_E&& expected,
                                       Value_A&& actual,
                                       Value_R rel)
{
    using VE = std::remove_cv_t<std::remove_reference_t<Value_E>>;
    using VA = std::remove_cv_t<std::remove_reference_t<Value_A>>;

    static_assert(can_soft_equiv<VE, VA>(),
                  "Invalid types for soft equivalence");

    // Construct with automatic or specified tolerances
    using ValueT = typename SoftPrecisionType<VE, VA>::type;
    using BinaryOp = EqualOr<SoftEqual<ValueT>>;

    return IsSoftEquivImpl(static_cast<ValueT>(expected),
                           expected_expr,
                           static_cast<ValueT>(actual),
                           actual_expr,
                           BinaryOp{static_cast<ValueT>(rel)});
}

//---------------------------------------------------------------------------//
// CONTAINER EQUIVALENCE
//---------------------------------------------------------------------------//
//! A single index/expected/actual value
template<class T1, class T2>
struct FailedValue
{
    using size_type = std::size_t;
    using first_type = T1;
    using second_type = T2;

    size_type index;
    first_type expected;
    second_type actual;
};

// Two Container Traits
template<class C1, class C2>
struct TCT
{
    template<class C>
    using value_type_ = typename ContTraits<C>::value_type;
    template<class C>
    using nc_value_type_ = typename std::remove_const<value_type_<C>>::type;

    using first_type = nc_value_type_<C1>;
    using second_type = nc_value_type_<C2>;

    using common_type =
        typename std::common_type<first_type, second_type>::type;

    using VecFailedValue = std::vector<FailedValue<first_type, second_type>>;
};

// Failed value iterator traits
template<class Iter1, class Iter2>
struct FVIT
{
    template<class I>
    using value_type_ = typename std::iterator_traits<I>::value_type;
    template<class I>
    using nc_value_type_ = typename std::remove_const<value_type_<I>>::type;

    using first_type = nc_value_type_<Iter1>;
    using second_type = nc_value_type_<Iter2>;

    using type = FailedValue<first_type, second_type>;
    using Vec_t = std::vector<type>;
};

//---------------------------------------------------------------------------//
/*!
 * Check if type T is a container.
 */
template<typename T, typename = void>
struct IsContainer : std::false_type
{
};

template<typename T>
struct IsContainer<T, std::void_t<typename T::const_iterator>> : std::true_type
{
};

template<typename T, std::size_t N>
struct IsContainer<T[N]> : std::true_type
{
};

//---------------------------------------------------------------------------//
/*!
 * Get the type of a container.
 */
template<class T, typename = void>
struct ValueType
{
    using type = typename T::value_type;
};

template<typename T, std::size_t N>
struct ValueType<T[N]>
{
    using type = T;
};

template<class T>
using ValueTypeT = typename ValueType<T>::type;

//---------------------------------------------------------------------------//
/*!
 * Recursively get the underlying scalar type of a container.
 */
template<typename T, typename = void>
struct ScalarValueType
{
    using type = T;
};

template<typename T>
struct ScalarValueType<T, std::enable_if_t<IsContainer<T>::value>>
{
    using type = typename ScalarValueType<ValueTypeT<T>>::type;
};

template<typename T>
using ScalarValueTypeT = typename ScalarValueType<T>::type;

//---------------------------------------------------------------------------//
/*!
 * Compare a range of values.
 */
template<class Iter1, class Iter2, class BinaryOp>
::testing::AssertionResult
IsRangeEqImpl(Iter1 e_iter,
              Iter1 e_end,
              char const* expected_expr,
              Iter2 a_iter,
              Iter2 a_end,
              char const* actual_expr,
              typename FVIT<Iter1, Iter2>::Vec_t& failures,
              BinaryOp comp)
{
    using size_type = std::size_t;
    size_type expected_size = std::distance(e_iter, e_end);
    size_type actual_size = std::distance(a_iter, a_end);

    // First, check that the sizes are equal
    if (expected_size != actual_size)
    {
        ::testing::AssertionResult failure = ::testing::AssertionFailure();

        failure << " Size of: " << actual_expr << "\n  Actual: " << actual_size
                << "\nExpected: " << expected_expr
                << ".size()\nWhich is: " << expected_size << '\n';
        return failure;
    }

    // Save start iterator in order to save index
    Iter1 const e_begin = e_iter;

    for (; e_iter != e_end; ++e_iter, ++a_iter)
    {
        if (!comp(*e_iter, *a_iter))
        {
            size_type i = e_iter - e_begin;
            failures.push_back({i, *e_iter, *a_iter});
        }
    }

    if (failures.empty())
    {
        return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult result = ::testing::AssertionFailure();
    result << "Values in: " << actual_expr << "\n Expected: " << expected_expr
           << '\n'
           << failures.size() << " of " << expected_size << " elements differ";
    if (failures.size() > 40)
    {
        result << " (truncating by removing all but the first and last 20)";
        failures.erase(failures.begin() + 20, failures.end() - 20);
    }
    result << '\n';
    return result;
}

//-------------------------------------------------------------------------//
/*!
 * Compare vectors with soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA, class BinaryOp>
::testing::AssertionResult IsVecSoftEquivImpl(ContainerE const& expected,
                                              char const* expected_expr,
                                              ContainerA const& actual,
                                              char const* actual_expr,
                                              BinaryOp comp)
{
    if constexpr (IsContainer<ValueTypeT<ContainerE>>::value
                  && IsContainer<ValueTypeT<ContainerA>>::value)
    {
        // Handle nested containers recursively
        auto exp_size = std::distance(std::begin(expected), std::end(expected));
        auto act_size = std::distance(std::begin(actual), std::end(actual));

        if (exp_size != act_size)
        {
            ::testing::AssertionResult failure = ::testing::AssertionFailure();

            failure << " Size of: " << actual_expr
                    << "\n  Actual: " << act_size
                    << "\nExpected: " << expected_expr
                    << ".size()\nWhich is: " << exp_size << '\n';
            return failure;
        }

        for (auto i : range(exp_size))
        {
            auto result = IsVecSoftEquivImpl(
                expected[i], expected_expr, actual[i], actual_expr, comp);
            if (!result)
            {
                return result;
            }
        }
        return ::testing::AssertionSuccess();
    }
    else
    {
        using Traits_t = TCT<ContainerE, ContainerA>;
        using Failed_t = FailedValue<typename Traits_t::first_type,
                                     typename Traits_t::second_type>;
        std::vector<Failed_t> failures;

        ::testing::AssertionResult result = IsRangeEqImpl(std::begin(expected),
                                                          std::end(expected),
                                                          expected_expr,
                                                          std::begin(actual),
                                                          std::end(actual),
                                                          actual_expr,
                                                          failures,
                                                          comp);

        if (!result)
        {
            if (failures.empty())
            {
                // Size was different; print the actual vector
                result << "Actual values: " << repr(actual) << ";\n";
            }
            else
            {
                // Inform user of failing tolerance
                result << "by " << comp.rel() << " relative error or "
                       << comp.abs() << " absolute error\n";
                // Print indices that were different
                result << float_failure_msg(
                    expected_expr, actual_expr, failures, comp.abs());
            }
        }
        return result;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print failure results.
 */
template<class T1, class T2>
std::string failure_msg(char const* expected_expr,
                        char const* actual_expr,
                        std::vector<FailedValue<T1, T2>> const& failures)
{
    using RT1 = ReprTraits<T1>;
    using RT2 = ReprTraits<T2>;
    using std::setw;

    // Calculate how many digits we need to space out
    int idig = num_digits(static_cast<unsigned long>(failures.back().index));
    int vdig = 16;

    // Construct our own stringstream because google test ignores setw
    std::ostringstream os;
    RT2::init(os);
    RT1::init(os);

    // Print column headers (unless expected/actual is too long)
    os << setw(idig) << 'i' << ' ' << setw(vdig)
       << trunc_string(vdig, expected_expr, "EXPECTED") << ' ' << setw(vdig)
       << trunc_string(vdig, actual_expr, "ACTUAL") << '\n';

    // Loop through failed indices and print values
    for (auto const& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig);
        RT1::print_value(os, f.expected);
        os << ' ' << setw(vdig);
        RT2::print_value(os, f.actual);
        os << '\n';
    }
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Print failure results for floating point comparisons.
 */
template<class T1, class T2>
std::string float_failure_msg(char const* expected_expr,
                              char const* actual_expr,
                              std::vector<FailedValue<T1, T2>> const& failures,
                              double abs_thresh)
{
    using std::setprecision;
    using std::setw;

    // Calculate how many digits we need to space out the index
    int idig = num_digits(static_cast<unsigned long>(failures.back().index));
    int vdig = std::max(std::numeric_limits<T1>::digits10,
                        std::numeric_limits<T2>::digits10);

    // Construct our own stringstream because google test ignores setw
    std::ostringstream os;
    os << setprecision(vdig);
    vdig += 4;

    // Try to use user-given expressions for headers, but fall back if the
    // column length is exceeded
    std::string e_expr(expected_expr);
    std::string a_expr(actual_expr);

    os << setw(idig) << 'i' << ' ' << setw(vdig)
       << trunc_string(vdig, expected_expr, "EXPECTED") << setw(vdig)
       << trunc_string(vdig, actual_expr, "ACTUAL") << setw(vdig)
       << "Difference" << '\n';

    // Loop through failed indices and print values
    for (auto const& f : failures)
    {
        os << setw(idig) << f.index << ' ' << setw(vdig) << f.expected << ' '
           << setw(vdig) << f.actual << ' ' << setw(vdig);

        if (std::isinf(f.expected))
        {
            os << "---";
        }
        else if (std::fabs(f.expected) > abs_thresh)
        {
            os << (f.actual - f.expected) / f.expected;
        }
        else
        {
            os << f.actual - f.expected;
        }
        os << '\n';
    }
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Print expected values.
 */
template<class Container>
void print_expected(Container const& data, std::string label)
{
    using RT = ReprTraits<Container>;
    using VRT = ReprTraits<typename RT::value_type>;

    std::cout << "static ";
    label.insert(0, "const expected_");
    VRT::print_type(std::cout, label.c_str());
    std::cout << "[] = ";

    std::ios orig_state(nullptr);
    orig_state.copyfmt(std::cout);
    VRT::init(std::cout);
    RT::print_value(std::cout, data);
    std::cout.copyfmt(orig_state);

    std::cout << ";\n";
}

//---------------------------------------------------------------------------//
/*!
 * Compare two containers.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecEq(char const* expected_expr,
                                   char const* actual_expr,
                                   ContainerE const& expected,
                                   ContainerA const& actual)
{
    if constexpr (IsContainer<ValueTypeT<ContainerE>>::value
                  && IsContainer<ValueTypeT<ContainerA>>::value)
    {
        // Handle nested containers recursively
        auto exp_size = std::distance(std::begin(expected), std::end(expected));
        auto act_size = std::distance(std::begin(actual), std::end(actual));

        if (exp_size != act_size)
        {
            ::testing::AssertionResult failure = ::testing::AssertionFailure();

            failure << " Size of: " << actual_expr
                    << "\n  Actual: " << act_size
                    << "\nExpected: " << expected_expr
                    << ".size()\nWhich is: " << exp_size << '\n';
            return failure;
        }

        for (auto i : range(exp_size))
        {
            auto result
                = IsVecEq(expected_expr, actual_expr, expected[i], actual[i]);
            if (!result)
            {
                return result;
            }
        }
        return ::testing::AssertionSuccess();
    }
    else
    {
        using Traits_t = TCT<ContainerE, ContainerA>;

        typename Traits_t::VecFailedValue failures;

        ::testing::AssertionResult result
            = IsRangeEqImpl(std::begin(expected),
                            std::end(expected),
                            expected_expr,
                            std::begin(actual),
                            std::end(actual),
                            actual_expr,
                            failures,
                            std::equal_to<typename Traits_t::common_type>());

        if (!result)
        {
            if (failures.empty())
            {
                // Size was different; print the actual vector
                result << "Actual values: " << repr(actual) << ";\n";
            }
            else
            {
                // Print indices that were different
                result << failure_msg(expected_expr, actual_expr, failures);
            }
        }

        return result;
    }
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 */
template<class ContainerE, class ContainerA>
::testing::AssertionResult IsVecSoftEquiv(char const* expected_expr,
                                          char const* actual_expr,
                                          ContainerE const& expected,
                                          ContainerA const& actual)
{
    using value_type_E = ScalarValueTypeT<ContainerE>;
    using value_type_A = ScalarValueTypeT<ContainerA>;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = EqualOr<SoftEqual<ValueT>>;

    // Construct with automatic or specified tolerances
    return IsVecSoftEquivImpl(
        expected, expected_expr, actual, actual_expr, BinaryOp());
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 *
 * This signature uses the default tolerance for the appropriate floating point
 * operations.
 */
template<class ContainerE, class ContainerA, class T>
::testing::AssertionResult IsVecSoftEquiv(char const* expected_expr,
                                          char const* actual_expr,
                                          char const*,
                                          ContainerE const& expected,
                                          ContainerA const& actual,
                                          T rel)
{
    using value_type_E = ScalarValueTypeT<ContainerE>;
    using value_type_A = ScalarValueTypeT<ContainerA>;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = EqualOr<SoftEqual<ValueT>>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(expected,
                              expected_expr,
                              actual,
                              actual_expr,
                              BinaryOp{static_cast<ValueT>(rel)});
}

//-------------------------------------------------------------------------//
/*!
 * Compare two containers using soft equivalence.
 *
 * Used by \c EXPECT_VEC_CLOSE.
 */
template<class ContainerE, class ContainerA, class T>
::testing::AssertionResult IsVecSoftEquiv(char const* expected_expr,
                                          char const* actual_expr,
                                          char const*,
                                          char const*,
                                          ContainerE const& expected,
                                          ContainerA const& actual,
                                          T rel,
                                          T abs)
{
    using value_type_E = ScalarValueTypeT<ContainerE>;
    using value_type_A = ScalarValueTypeT<ContainerA>;

    static_assert(can_soft_equiv<value_type_E, value_type_A>(),
                  "Invalid types for soft equivalence");

    using ValueT = typename SoftPrecisionType<value_type_E, value_type_A>::type;
    using BinaryOp = EqualOr<SoftEqual<ValueT>>;

    // Construct with given tolerance
    return IsVecSoftEquivImpl(
        expected,
        expected_expr,
        actual,
        actual_expr,
        BinaryOp{static_cast<ValueT>(rel), static_cast<ValueT>(abs)});
}

//---------------------------------------------------------------------------//
/*!
 * Compare two JSON objects.
 */
::testing::AssertionResult IsJsonEq(char const* expected_expr,
                                    char const* actual_expr,
                                    std::string_view expected,
                                    std::string_view actual);

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
