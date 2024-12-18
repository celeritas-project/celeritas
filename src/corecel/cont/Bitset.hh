//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Bitset.hh
//---------------------------------------------------------------------------//
#pragma once

#include <climits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/detail/BitsetUtils.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device-compatible bitset implementation.
 *
 * This implementation is based on libstdc++'s std::bitset implementation.
 * It it a subset of the C++ standard, but it should be sufficient
 * for our current use case. Given that GPU typically use 32-bit words, this
 * uses unsigned int as the word type instead of the unsigned long used by the
 * standard library. This container is not thread-safe, multiple threads are
 * likely to manipulate the same word and access is not synchronized.
 *
 * The following methods are not implemented:
 * - conversions to string, to_ulong, to_ullong
 * - set operations with other bitsets
 * - stream operators
 * - hash support
 * - construct from string, from_ulong, from_ullong
 */
template<size_type N>
class Bitset
{
  public:
    //!@{
    //! \name Type aliases
    using word_type = unsigned int;
    //!@}

    static constexpr size_type bits_per_word = CHAR_BIT * sizeof(word_type);
    static constexpr size_type num_words = (N / bits_per_word)
                                           + (N % bits_per_word == 0 ? 0 : 1);

  public:
    class reference;

    CELER_CONSTEXPR_FUNCTION bool operator==(Bitset const& other) const noexcept
    {
        for (size_type i = 0; i < num_words; ++i)
        {
            if (words_[i] != other.words_[i])
            {
                return false;
            }
        }
        return true;
    }

    CELER_CONSTEXPR_FUNCTION bool operator!=(Bitset const& other) const noexcept
    {
        return !(*this == other);
    }

    CELER_CONSTEXPR_FUNCTION bool operator[](size_type pos) const noexcept
    {
        return (this->get_word(pos) & Bitset::mask(pos))
               != static_cast<word_type>(0);
    }

    CELER_CONSTEXPR_FUNCTION reference operator[](size_type pos) noexcept
    {
        return reference(*this, pos);
    }

    CELER_CONSTEXPR_FUNCTION bool test(size_type pos) const
        noexcept(!CELERITAS_DEBUG)
    {
        CELER_EXPECT(pos < N);
        return (*this)[pos];
    }

    CELER_CONSTEXPR_FUNCTION bool all() const noexcept
    {
        for (size_type i = 0; i < num_words - 1; ++i)
        {
            if (words_[i] != ~static_cast<word_type>(0))
            {
                return false;
            }
        }
        // only compare the last word up to the last bit of the bitset
        return this->last_word()
               == (~static_cast<word_type>(0)
                   >> (num_words * bits_per_word - N));
    }

    CELER_CONSTEXPR_FUNCTION bool any() const noexcept
    {
        for (size_type i = 0; i < num_words; ++i)
        {
            if (words_[i] != static_cast<word_type>(0))
            {
                return true;
            }
        }

        return false;
    }

    CELER_CONSTEXPR_FUNCTION bool none() const noexcept
    {
        return !this->any();
    }

    CELER_CONSTEXPR_FUNCTION size_type count() const noexcept
    {
        size_type count = 0;
        for (size_type i = 0; i < num_words; ++i)
        {
            count += celeritas::popcount(words_[i]);
        }

        return count;
    }

    CELER_CONSTEXPR_FUNCTION size_type size() const noexcept { return N; }

    CELER_CONSTEXPR_FUNCTION Bitset& set() noexcept
    {
        for (size_type i = 0; i < num_words; ++i)
        {
            words_[i] = ~static_cast<word_type>(0);
        }
        // sanitize the last word
        detail::Sanitize<N % bits_per_word>::sanitize(this->last_word());
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION Bitset&
    set(size_type pos, bool value = true) noexcept
    {
        if (value)
        {
            this->get_word(pos) |= Bitset::mask(pos);
        }
        else
        {
            this->get_word(pos) &= ~Bitset::mask(pos);
        }

        return *this;
    }

    CELER_CONSTEXPR_FUNCTION Bitset& reset() noexcept
    {
        for (size_type i = 0; i < num_words; ++i)
        {
            words_[i] = static_cast<word_type>(0);
        }

        return *this;
    }

    CELER_CONSTEXPR_FUNCTION Bitset& reset(size_type pos) noexcept
    {
        this->get_word(pos) &= ~Bitset::mask(pos);
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION Bitset& flip() noexcept
    {
        for (size_type i = 0; i < num_words; ++i)
        {
            words_[i] = ~words_[i];
        }
        // sanitize the last word
        detail::Sanitize<N % bits_per_word>::sanitize(last_word());
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION Bitset& flip(size_type pos) noexcept
    {
        this->get_word(pos) ^= Bitset::mask(pos);
        return *this;
    }

  private:
    static CELER_CONSTEXPR_FUNCTION size_type which_word(size_type pos) noexcept
    {
        return pos / bits_per_word;
    }

    static CELER_CONSTEXPR_FUNCTION size_type which_bit(size_type pos) noexcept
    {
        return pos % bits_per_word;
    }

    static CELER_CONSTEXPR_FUNCTION word_type mask(size_type pos) noexcept
    {
        return static_cast<word_type>(1) << Bitset::which_bit(pos);
    }

    CELER_CONSTEXPR_FUNCTION word_type get_word(size_type pos) const noexcept
    {
        return words_[Bitset::which_word(pos)];
    }

    CELER_CONSTEXPR_FUNCTION word_type& get_word(size_type pos) noexcept
    {
        return words_[Bitset::which_word(pos)];
    }

    CELER_CONSTEXPR_FUNCTION word_type& last_word() noexcept
    {
        return words_[num_words - 1];
    }

    CELER_CONSTEXPR_FUNCTION word_type last_word() const noexcept
    {
        return words_[num_words - 1];
    }

    word_type words_[num_words] = {};
};

/*!
 * Reference to a single bit in the bitset.
 * This is used to implement the mutable operator[].
 */
template<size_type N>
class Bitset<N>::reference
{
    friend class Bitset;

  public:
    CELER_CONSTEXPR_FUNCTION
    reference(Bitset& b, size_type pos) noexcept
        : word_pointer_(&b.get_word(pos)), bit_pos_(Bitset::which_bit(pos))
    {
    }

    CELER_CONSTEXPR_FUNCTION reference(reference const&) = default;

    CELER_FUNCTION ~reference() noexcept = default;

    // For b[i] = x;
    CELER_CONSTEXPR_FUNCTION
    reference& operator=(bool x) noexcept
    {
        if (x)
        {
            *word_pointer_ |= Bitset::mask(bit_pos_);
        }
        else
        {
            *word_pointer_ &= ~Bitset::mask(bit_pos_);
        }
        return *this;
    }

    // For b[i] = b[j];
    CELER_CONSTEXPR_FUNCTION
    reference& operator=(reference const& j) noexcept
    {
        if (this != &j)
        {
            if (*j.word_pointer_ & Bitset::mask(j.bit_pos_))
            {
                *word_pointer_ |= Bitset::mask(bit_pos_);
            }
            else
            {
                *word_pointer_ &= ~Bitset::mask(bit_pos_);
            }
        }
        return *this;
    }

    // Flips the bit
    CELER_CONSTEXPR_FUNCTION
    bool operator~() const noexcept { return !static_cast<bool>(*this); }

    // For x = b[i];
    CELER_CONSTEXPR_FUNCTION
    operator bool() const noexcept
    {
        return (*word_pointer_ & Bitset::mask(bit_pos_)) != 0;
    }

    // For b[i].flip();
    CELER_CONSTEXPR_FUNCTION
    reference& flip() noexcept
    {
        *word_pointer_ ^= Bitset::mask(bit_pos_);
        return *this;
    }

  private:
    word_type* word_pointer_{nullptr};
    size_type bit_pos_{0};
};

}  // namespace celeritas
