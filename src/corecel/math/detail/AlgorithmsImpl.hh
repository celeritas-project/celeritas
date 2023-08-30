//----------------------------------*-C++-*----------------------------------//
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//---------------------------------------------------------------------------//
/*!
 * \file corecel/math/detail/AlgorithmsImpl.hh
 * \brief Algorithm implementations directly derived from LLVM libc++
 */
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Forward declare cuda-compatible swap/move
template<class T>
CELER_FORCEINLINE_FUNCTION void trivial_swap(T&, T&) noexcept;

namespace detail
{
//---------------------------------------------------------------------------//
template<class RandomAccessIt>
using difference_type_t =
    typename std::iterator_traits<RandomAccessIt>::difference_type;

//---------------------------------------------------------------------------//
// LOWER and UPPER BOUNDS
//---------------------------------------------------------------------------//
//!@{
/*!
 * Perform division-by-two quickly for positive integers.
 *
 * See llvm.org/PR39129.
 */
template<typename Integral>
CELER_CONSTEXPR_FUNCTION
    typename std::enable_if<std::is_integral<Integral>::value, Integral>::type
    half_positive(Integral value)
{
    return static_cast<Integral>(
        static_cast<typename std::make_unsigned<Integral>::type>(value) / 2);
}

template<typename T>
CELER_CONSTEXPR_FUNCTION
    typename std::enable_if<!std::is_integral<T>::value, T>::type
    half_positive(T value)
{
    return value / 2;
}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Implementation of binary search lower-bound assuming iterator arithmetic.
 */
template<class Compare, class ForwardIterator, class T>
CELER_FUNCTION ForwardIterator lower_bound_impl(ForwardIterator first,
                                                ForwardIterator last,
                                                T const& value_,
                                                Compare comp)
{
    using difference_type = difference_type_t<ForwardIterator>;

    difference_type len = last - first;
    while (len != 0)
    {
        difference_type half_len = ::celeritas::detail::half_positive(len);
        ForwardIterator m = first + half_len;
        if (comp(*m, value_))
        {
            first = ++m;
            len -= half_len + 1;
        }
        else
            len = half_len;
    }
    return first;
}

//---------------------------------------------------------------------------//
/*!
 * Implementation of linear search lower-bound assuming iterator arithmetic.
 */
template<class Compare, class ForwardIterator, class T>
CELER_FUNCTION ForwardIterator lower_bound_linear_impl(ForwardIterator first,
                                                       ForwardIterator last,
                                                       T const& value_,
                                                       Compare comp)
{
    for (ForwardIterator it = first; it != last; ++it)
    {
        if (!comp(*it, value_))
        {
            return it;
        }
    }

    return last;
}

//---------------------------------------------------------------------------//
/*!
 * Implementation of upper-bound assuming iterator arithmetic.
 */
template<class Compare, class ForwardIterator, class T>
CELER_FUNCTION ForwardIterator upper_bound_impl(ForwardIterator first,
                                                ForwardIterator last,
                                                T const& value_,
                                                Compare comp)
{
    using difference_type = difference_type_t<ForwardIterator>;

    difference_type len = last - first;
    while (len != 0)
    {
        difference_type half_len = ::celeritas::detail::half_positive(len);
        ForwardIterator m = first + half_len;
        if (comp(value_, *m))
        {
            len = half_len;
        }
        else
        {
            first = ++m;
            len -= half_len + 1;
        }
    }
    return first;
}

//---------------------------------------------------------------------------//
// PARTITION
//---------------------------------------------------------------------------//
/*!
 * Partition elements in the given range, "true" before "false".
 *
 * This implementation requires bidirectional iterators (typically true since
 * celeritas tends to use contiguous data).
 */
template<class Predicate, class BidirectionalIterator>
CELER_FUNCTION BidirectionalIterator partition_impl(BidirectionalIterator first,
                                                    BidirectionalIterator last,
                                                    Predicate pred)
{
    while (true)
    {
        while (true)
        {
            if (first == last)
                return first;
            if (!pred(*first))
                break;
            ++first;
        }
        do
        {
            if (first == --last)
                return first;
        } while (!pred(*last));
        trivial_swap(*first, *last);
        ++first;
    }
}

//---------------------------------------------------------------------------//
// SORT
//---------------------------------------------------------------------------//
/*!
 * Cast a value to an rvalue reference.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION auto trivial_move(T&& v) noexcept ->
    typename std::remove_reference<T>::type&&
{
    return static_cast<typename std::remove_reference<T>::type&&>(v);
}

//---------------------------------------------------------------------------//
/*!
 * Move the top element of the heap to its properly ordered place.
 *
 * The reverse ordering of the template parameters here is to allow the public
 * functions (sort, partial_sort) to explicitly pass references to the
 * comparator rather than passing by value every time.
 */
template<class Compare, class RandomAccessIt>
CELER_FUNCTION void sift_down(RandomAccessIt first,
                              RandomAccessIt,
                              Compare comp,
                              difference_type_t<RandomAccessIt> len,
                              RandomAccessIt start)
{
    using difference_type = difference_type_t<RandomAccessIt>;
    using value_type =
        typename std::iterator_traits<RandomAccessIt>::value_type;

    // Left-child of start is at 2 * start + 1
    // Right-child of start is at 2 * start + 2
    difference_type child = start - first;

    if (len < 2 || (len - 2) / 2 < child)
    {
        return;
    }

    child = 2 * child + 1;
    RandomAccessIt child_i = first + child;

    if ((child + 1) < len && comp(*child_i, *(child_i + difference_type(1))))
    {
        // Right-child exists and is greater than left-child
        ++child_i;
        ++child;
    }

    if (comp(*child_i, *start))
    {
        // We are in heap order: start is larger than its largest child
        return;
    }

    value_type top(trivial_move(*start));
    do
    {
        // We are not in heap order: swap the parent with its largest child
        *start = trivial_move(*child_i);
        start = child_i;

        if ((len - 2) / 2 < child)
            break;

        // Recompute the child based off of the updated parent
        child = 2 * child + 1;
        child_i = first + child;

        if ((child + 1) < len
            && comp(*child_i, *(child_i + difference_type(1))))
        {
            // Right-child exists and is greater than left-child
            ++child_i;
            ++child;
        }
    } while (!comp(*child_i, top));
    *start = trivial_move(top);
}

//---------------------------------------------------------------------------//
/*!
 * Move the largest (first) element in a heap to the last position.
 */
template<class Compare, class RandomAccessIt>
CELER_FORCEINLINE_FUNCTION void pop_heap(RandomAccessIt first,
                                         RandomAccessIt last,
                                         Compare comp,
                                         difference_type_t<RandomAccessIt> len)
{
    if (len > 1)
    {
        trivial_swap(*first, *--last);
        ::celeritas::detail::sift_down<Compare>(
            first, last, comp, len - 1, first);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert the given range to a heap.
 */
template<class Compare, class RandomAccessIt>
CELER_FUNCTION void
make_heap(RandomAccessIt first, RandomAccessIt last, Compare comp)
{
    using difference_type = difference_type_t<RandomAccessIt>;

    difference_type n = last - first;
    if (n > 1)
    {
        for (difference_type start = (n - 2) / 2; start >= 0; --start)
        {
            ::celeritas::detail::sift_down<Compare>(
                first, last, comp, n, first + start);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a heap to a sorted range.
 *
 * The \c void cast shouldn't ever be something we have to worry about, but
 * it's left in there from the LLVM implementation "to handle evil iterators
 * that overload operator comma" (bd7c7b55511a4b4b50b77559a44eff6d350224c4).
 */
template<class Compare, class RandomAccessIt>
CELER_FUNCTION void
sort_heap(RandomAccessIt first, RandomAccessIt last, Compare comp)
{
    using difference_type = difference_type_t<RandomAccessIt>;

    for (difference_type n = last - first; n > 1; --last, (void)--n)
    {
        ::celeritas::detail::pop_heap<Compare>(first, last, comp, n);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sort the first elements of the range.
 *
 * This uses the remaining elements of the range [middle, last) as swap space.
 */
template<class Compare, class RandomAccessIt>
CELER_FUNCTION void partial_sort(RandomAccessIt first,
                                 RandomAccessIt middle,
                                 RandomAccessIt last,
                                 Compare comp)
{
    using difference_type = difference_type_t<RandomAccessIt>;

    ::celeritas::detail::make_heap<Compare>(first, middle, comp);

    difference_type len = middle - first;
    for (RandomAccessIt i = middle; i != last; ++i)
    {
        if (comp(*i, *first))
        {
            trivial_swap(*i, *first);
            ::celeritas::detail::sift_down<Compare>(
                first, middle, comp, len, first);
        }
    }
    ::celeritas::detail::sort_heap<Compare>(first, middle, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a heap sort on the given range.
 *
 * The heap sort algorithm converts the unsorted array into an ordered
 * contiguous heap in-place, then converts the heap back to a sorted array. We
 * choose heapsort because it requires no dynamic allocation and has both best
 * and worst case time of O(N log N).
 *
 * The heap sort algorithm here is derived almost verbatim from the LLVM
 * libc++.
 */
template<class Compare, class RandomAccessIt>
CELER_FUNCTION void
heapsort_impl(RandomAccessIt first, RandomAccessIt last, Compare comp)
{
    ::celeritas::detail::partial_sort<Compare>(first, last, last, comp);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
