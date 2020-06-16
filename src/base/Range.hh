//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// Range utility function.
//---------------------------------------------------------------------------//

#ifndef celeritas_Range_hh
#define celeritas_Range_hh

#include <iterator>
#include <type_traits>
#include <utility>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \fn range
 * \tparam T Value type to iterate over
 * \brief Get iterators over a range of values, or a semi-infinite range.
 *
 * \par Code Sample:
 * \code

    for (auto i : Range(1, 5))
        cout << i << "\n";

    // Range of [0, 10)
    for (auto u : Range(10u))
        cout << u << "\n";

    for (auto c : Range('a', 'd'))
        cout << c << "\n";

    for (auto i : Count(100).step(-3))
        if (i < 90) break;
        else        cout << i << "\n";

 * \endcode
 */
//---------------------------------------------------------------------------//

namespace internal
{
//---------------------------------------------------------------------------//
template<typename T, typename Enable = void>
struct range_type_traits
{
    using value_type   = T;
    using counter_type = T;
};

template<typename T>
struct range_type_traits<T, typename std::enable_if<std::is_enum<T>::value>::type>
{
    using value_type   = T;
    using counter_type = typename std::underlying_type<T>::type;
};

//---------------------------------------------------------------------------//
/*!
 * \class range_iter
 */
template<typename T>
class range_iter : public std::iterator<std::input_iterator_tag, T>
{
  public:
    //@{
    //! Type aliases
    using traits_t     = range_type_traits<T>;
    using value_type   = typename traits_t::value_type;
    using counter_type = typename traits_t::counter_type;
    //@}

  protected:
    counter_type value_;

  public:
    // >>> CONSTRUCTOR

    range_iter(value_type value) : value_(static_cast<counter_type>(value)) {}

    // >>> ACCESSORS

    value_type operator*() const { return static_cast<value_type>(value_); }

    value_type const* operator->() const
    {
        return static_cast<const value_type*>(&value_);
    }

    // >>> ARITHMETIC

    range_iter& operator++()
    {
        ++value_;
        return *this;
    }

    range_iter operator++(int)
    {
        auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(range_iter const& other) const
    {
        return value_ == other.value_;
    }

    bool operator!=(range_iter const& other) const
    {
        return !(*this == other);
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class inf_range_iter
 *
 * Iterator that never finishes.
 */
template<typename T>
class inf_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;

  public:
    inf_range_iter(T value_ = T()) : Base(value_) {}

    bool operator==(inf_range_iter const&) const { return false; }
    bool operator!=(inf_range_iter const&) const { return true; }
};

//---------------------------------------------------------------------------//
/*!
 * \class step_range_iter
 */
template<typename T>
class step_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;

  protected:
    using Base::value_;
    T step_;

  public:
    step_range_iter(T value, T step) : Base(value), step_(step) {}

    step_range_iter& operator++()
    {
        value_ += step_;
        return *this;
    }

    step_range_iter operator++(int)
    {
        auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(step_range_iter const& other) const
    {
        return step_ > 0 ? value_ >= other.value_ : value_ < other.value_;
    }

    bool operator!=(step_range_iter const& other) const
    {
        return !(*this == other);
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class inf_step_range_iter
 *
 * Iterator that never finishes.
 */
template<typename T>
class inf_step_range_iter : public step_range_iter<T>
{
    using Base = step_range_iter<T>;

  public:
    inf_step_range_iter(T current = T(), T step = T()) : Base(current, step) {}

    bool operator==(inf_step_range_iter const&) const { return false; }
    bool operator!=(inf_step_range_iter const&) const { return true; }
};

} // namespace internal

//---------------------------------------------------------------------------//
/*!
 * \class StepRange
 *
 * Proxy container for iterating over a range of integral values with a step
 * between their values.
 */
template<typename T>
class StepRange
{
  public:
    using IterT = internal::step_range_iter<T>;

    StepRange(T begin, T end, T step) : begin_(begin, step), end_(end, step) {}

    IterT begin() const { return begin_; }
    IterT end() const { return end_; }

  private:
    IterT begin_;
    IterT end_;
};

//---------------------------------------------------------------------------//
/*!
 * \class InfStepRange
 *
 * Proxy iterator for iterating over a range of integral values with a step
 * between their values.
 */
template<typename T>
class InfStepRange
{
  public:
    using IterT = internal::inf_step_range_iter<T>;

    //! Construct from start/stop
    InfStepRange(T begin, T step) : begin_(begin, step) {}

    IterT begin() const { return begin_; }
    IterT end() const { return IterT(); }

  private:
    IterT begin_;
};

//---------------------------------------------------------------------------//
/*!
 * \class FiniteRange
 *
 * Proxy container for iterating over a range of integral values.
 */
template<typename T>
class FiniteRange
{
  public:
    using IterT        = internal::range_iter<T>;
    using counter_type = typename internal::range_type_traits<T>::counter_type;

    //! Empty constructor for empty range
    FiniteRange() : begin_(T()), end_(T()) {}

    //! Construct from start/stop
    FiniteRange(T begin, T end) : begin_(begin), end_(end) {}

    //! Return a stepped range using a different integer type
    template<typename U>
    StepRange<typename std::common_type<T, U>::type> step(U step)
    {
        if (step < 0)
        {
            // NOTE: if T and U are not compatible (e.g. T is unsigned and U is
            // signed) then this should raise an implicit conversion error .
            return {*end_ + step, *begin_, step};
        }

        return {*begin_, *end_, step};
    }

    IterT        begin() const { return begin_; }
    IterT        end() const { return end_; }
    counter_type size() const { return *end_ - *begin_; }
    bool         empty() const { return end_ == begin_; }

  private:
    IterT begin_;
    IterT end_;
};

//---------------------------------------------------------------------------//
/*!
 * \class InfiniteRange
 *
 * Proxy container for iterating over a range of integral values.
 */
template<typename T>
class InfiniteRange
{
  public:
    using IterT = internal::inf_range_iter<T>;

    InfiniteRange(T begin) : begin_(begin) {}

    InfStepRange<T> step(T step) { return {*begin_, step}; }

    IterT begin() const { return begin_; }
    IterT end() const { return IterT(); }
    bool  empty() const { return false; }

  private:
    IterT begin_;
};

//---------------------------------------------------------------------------//
/*!
 * \fn Range
 *
 * Return a range over fixed beginning and end values.
 */
template<typename T>
FiniteRange<T> range(T begin, T end)
{
    return {begin, end};
}

//---------------------------------------------------------------------------//
/*!
 * \fn Range
 *
 * Return a range over a pair of beginning and ending values.
 */
template<typename T>
FiniteRange<T> range(std::pair<T, T> begin_end)
{
    return {begin_end.first, begin_end.second};
}

//---------------------------------------------------------------------------//
/*!
 * \fn Range
 *
 * Return a range with the default start value (0 for numeric types)
 */
template<typename T>
FiniteRange<T> range(T end)
{
    return {T(), end};
}

//---------------------------------------------------------------------------//
/*!
 * \fn Count
 *
 * Count upward from zero.
 */
template<typename T>
InfiniteRange<T> count()
{
    return {T()};
}

//---------------------------------------------------------------------------//
/*!
 * \fn Count
 *
 * Count upward from a value.
 */
template<typename T>
InfiniteRange<T> count(T begin)
{
    return {begin};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
#endif // celeritas_Range_hh
