//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Bitset.hh
//---------------------------------------------------------------------------//
#pragma once

#include <climits>
#include <cstdint>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
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
 * likely to manipulate the same word, even when accessing different indices.
 *
 * The following methods are not implemented:
 * - conversions to string, to_ulong, to_ullong
 * - stream operators
 * - shift operators
 * - hash support
 * - construct from string, from_ullong
 */
template<size_type N>
class Bitset
{
    static_assert(N > 0, "Zero-sized bitset is not supported");

  public:
    //!@{
    //! \name Type aliases
    using word_type = std::conditional_t<
        (N <= 8),
        std::uint8_t,
        std::conditional_t<(N <= 16), std::uint16_t, size_type>>;
    //!@}

    class reference;

  public:
    //// CONSTRUCTORS ////

    // Default construct with zeros for all bits
    constexpr Bitset() = default;

    // Construct implicitly from a bitset encoded as an integer
    CELER_CONSTEXPR_FUNCTION Bitset(word_type value) noexcept;

    //// COMPARISON ////

    // Test equality with another bitset
    CELER_CONSTEXPR_FUNCTION bool
    operator==(Bitset const& other) const noexcept;

    // Test equality with another bitset
    CELER_CONSTEXPR_FUNCTION bool
    operator!=(Bitset const& other) const noexcept;

    //// ACCESSORS ////

    // Access to a single bit
    CELER_CONSTEXPR_FUNCTION bool operator[](size_type pos) const
        noexcept(!CELERITAS_DEBUG);

    // Mutable access to a single bit
    CELER_CONSTEXPR_FUNCTION reference
    operator[](size_type pos) noexcept(!CELERITAS_DEBUG);

    // Check if all bits are set
    CELER_CONSTEXPR_FUNCTION bool all() const noexcept;

    // Check if any bits are set
    CELER_CONSTEXPR_FUNCTION bool any() const noexcept;

    // Check if no bits are set
    CELER_CONSTEXPR_FUNCTION bool none() const noexcept;

    // Number of bits set to true
    CELER_CONSTEXPR_FUNCTION size_type count() const noexcept;

    //! Number of bits in the bitset
    CELER_CONSTEXPR_FUNCTION size_type size() const noexcept { return N; }

    //// MUTATORS ////

    // Bitwise AND with another bitset
    CELER_CONSTEXPR_FUNCTION Bitset& operator&=(Bitset const& other) noexcept;

    // Bitwise OR with another bitset
    CELER_CONSTEXPR_FUNCTION Bitset& operator|=(Bitset const& other) noexcept;

    // Bitwise XOR with another bitset
    CELER_CONSTEXPR_FUNCTION Bitset& operator^=(Bitset const& other) noexcept;

    // Return a copy with all bits flipped (binary NOT)
    CELER_CONSTEXPR_FUNCTION Bitset operator~() const noexcept;

    // Set all bits
    CELER_CONSTEXPR_FUNCTION Bitset& set() noexcept;

    // Set the bit at position pos
    CELER_CONSTEXPR_FUNCTION Bitset&
    set(size_type pos, bool value = true) noexcept(!CELERITAS_DEBUG);

    // Reset all bits
    CELER_CONSTEXPR_FUNCTION Bitset& reset() noexcept;

    // Reset the bit at position pos
    CELER_CONSTEXPR_FUNCTION Bitset&
    reset(size_type pos) noexcept(!CELERITAS_DEBUG);

    // Flip all bits
    CELER_CONSTEXPR_FUNCTION Bitset& flip() noexcept;

    // Flip the bit at position pos
    CELER_CONSTEXPR_FUNCTION Bitset&
    flip(size_type pos) noexcept(!CELERITAS_DEBUG);

  private:
    //// CONSTANTS ////

    static constexpr size_type bits_per_word_ = CHAR_BIT * sizeof(word_type);
    static constexpr size_type num_words_ = ceil_div(N, bits_per_word_);

    //// DATA ////

    //! Storage, default-initialized to zero
    word_type words_[num_words_]{};

    //// HELPER FUNCTIONS ////

    // Find the word index for a given bit position
    static CELER_CONSTEXPR_FUNCTION size_type which_word(size_type pos) noexcept;
    // Find the bit index in a word
    static CELER_CONSTEXPR_FUNCTION size_type which_bit(size_type pos) noexcept;

    // Create a mask for a given bit index
    static CELER_CONSTEXPR_FUNCTION word_type mask(size_type pos) noexcept;

    // Create a negative mask for a given bit index
    static CELER_CONSTEXPR_FUNCTION word_type neg_mask(size_type pos) noexcept;

    // Get the word for a given bit position
    CELER_CONSTEXPR_FUNCTION word_type get_word(size_type pos) const
        noexcept(!CELERITAS_DEBUG);

    // Get the word for a given bit position
    CELER_CONSTEXPR_FUNCTION word_type&
    get_word(size_type pos) noexcept(!CELERITAS_DEBUG);

    // Get the last word of the bitset
    CELER_CONSTEXPR_FUNCTION word_type& last_word() noexcept;

    // Get the last word of the bitset
    CELER_CONSTEXPR_FUNCTION word_type last_word() const noexcept;

    // Clear unused bits from the last word
    CELER_CONSTEXPR_FUNCTION void sanitize() noexcept;
};

//---------------------------------------------------------------------------//
// Bitset::reference definitions
//---------------------------------------------------------------------------//
/*!
 * Reference to a single bit in the bitset.
 *
 * This is used to implement the mutable operator[].
 */
template<size_type N>
class Bitset<N>::reference
{
  public:
    CELER_CONSTEXPR_FUNCTION
    reference(Bitset& b, size_type pos) noexcept
        : word_pointer_(&b.get_word(pos)), bit_pos_(Bitset::which_bit(pos))
    {
    }

    constexpr reference(reference const&) = default;

    ~reference() noexcept = default;

    //! Assignment for b[i] = x;
    CELER_CONSTEXPR_FUNCTION
    reference& operator=(bool x) noexcept
    {
        if (x)
        {
            *word_pointer_ |= Bitset::mask(bit_pos_);
        }
        else
        {
            *word_pointer_ &= Bitset::neg_mask(bit_pos_);
        }
        return *this;
    }

    //! Assignment for b[i] = b[j];
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
                *word_pointer_ &= Bitset::neg_mask(bit_pos_);
            }
        }
        return *this;
    }

    //! Flips the bit
    CELER_CONSTEXPR_FUNCTION
    bool operator~() const noexcept { return !static_cast<bool>(*this); }

    //! Conversion for bool x = b[i];
    CELER_CONSTEXPR_FUNCTION
    operator bool() const noexcept
    {
        return (*word_pointer_ & Bitset::mask(bit_pos_)) != 0;
    }

    //! To support b[i].flip();
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

//---------------------------------------------------------------------------//
// INLINE MEMBER FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
//! Construct from a word value
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>::Bitset(word_type value) noexcept
    : words_{value}
{
    if constexpr (num_words_ == 1)
    {
        this->sanitize();
    }
}

//---------------------------------------------------------------------------//
//! Compare for equality
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool
Bitset<N>::operator==(Bitset const& other) const noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        if (words_[i] != other.words_[i])
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
//! Compare for inequality
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool
Bitset<N>::operator!=(Bitset const& other) const noexcept
{
    return !(*this == other);
}

//---------------------------------------------------------------------------//
//! Access a single bit
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool Bitset<N>::operator[](size_type pos) const
    noexcept(!CELERITAS_DEBUG)
{
    CELER_EXPECT(pos < N);
    return (this->get_word(pos) & Bitset::mask(pos))
           != static_cast<word_type>(0);
}

//---------------------------------------------------------------------------//
//! Mutable access to a single bit
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto
Bitset<N>::operator[](size_type pos) noexcept(!CELERITAS_DEBUG) -> reference
{
    CELER_EXPECT(pos < N);
    return reference(*this, pos);
}

//---------------------------------------------------------------------------//
//! Check if all bits are set
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool Bitset<N>::all() const noexcept
{
    for (size_type i = 0; i < num_words_ - 1; ++i)
    {
        if (words_[i] != static_cast<word_type>(~word_type(0)))
        {
            return false;
        }
    }

    // Only compare the last word up to the last bit of the bitset
    return this->last_word()
           == (static_cast<word_type>(~word_type(0))
               >> (num_words_ * bits_per_word_ - N));
}

//---------------------------------------------------------------------------//
//! Check if any bits are set
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool Bitset<N>::any() const noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        if (words_[i] != word_type(0))
        {
            return true;
        }
    }

    return false;
}

//---------------------------------------------------------------------------//
//! Check if no bits are set
template<size_type N>
CELER_CONSTEXPR_FUNCTION bool Bitset<N>::none() const noexcept
{
    return !this->any();
}

//---------------------------------------------------------------------------//
//! Number of bits set to true
template<size_type N>
CELER_CONSTEXPR_FUNCTION size_type Bitset<N>::count() const noexcept
{
    size_type count = 0;
    for (size_type i = 0; i < num_words_; ++i)
    {
        count += celeritas::popcount(words_[i]);
    }

    return count;
}

//---------------------------------------------------------------------------//
//! Bitwise AND with another bitset
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::operator&=(Bitset const& other) noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] &= other.words_[i];
    }
    return *this;
}

//---------------------------------------------------------------------------//
//! Bitwise OR with another bitset
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::operator|=(Bitset const& other) noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] |= other.words_[i];
    }
    return *this;
}

//---------------------------------------------------------------------------//
//! Bitwise XOR with another bitset
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::operator^=(Bitset const& other) noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] ^= other.words_[i];
    }
    return *this;
}

//---------------------------------------------------------------------------//
//! Return a copy with all bits flipped (binary NOT)
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N> Bitset<N>::operator~() const noexcept
{
    return Bitset{*this}.flip();
}

//---------------------------------------------------------------------------//
//! Set all bits
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>& Bitset<N>::set() noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] = static_cast<word_type>(~word_type(0));
    }

    // Clear unused bits on the last word
    this->sanitize();

    return *this;
}

//---------------------------------------------------------------------------//
//! Set the bit at position pos
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::set(size_type pos, bool value) noexcept(!CELERITAS_DEBUG)
{
    CELER_EXPECT(pos < N);
    (*this)[pos] = value;
    return *this;
}

//---------------------------------------------------------------------------//
//! Reset all bits
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>& Bitset<N>::reset() noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] = word_type(0);
    }

    return *this;
}

//---------------------------------------------------------------------------//
//! Reset the bit at position pos
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::reset(size_type pos) noexcept(!CELERITAS_DEBUG)
{
    CELER_EXPECT(pos < N);
    this->get_word(pos) &= Bitset::neg_mask(pos);
    return *this;
}

//---------------------------------------------------------------------------//
//! Flip all bits
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>& Bitset<N>::flip() noexcept
{
    for (size_type i = 0; i < num_words_; ++i)
    {
        words_[i] = ~words_[i];
    }

    // Clear unused bits on the last word
    this->sanitize();

    return *this;
}

//---------------------------------------------------------------------------//
//! Flip the bit at position pos
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>&
Bitset<N>::flip(size_type pos) noexcept(!CELERITAS_DEBUG)
{
    CELER_EXPECT(pos < N);
    this->get_word(pos) ^= Bitset::mask(pos);
    return *this;
}

//---------------------------------------------------------------------------//
//! Find the word index for a given bit position
template<size_type N>
CELER_CONSTEXPR_FUNCTION size_type Bitset<N>::which_word(size_type pos) noexcept
{
    return pos / bits_per_word_;
}

//---------------------------------------------------------------------------//
//! Find the bit index in a word
template<size_type N>
CELER_CONSTEXPR_FUNCTION size_type Bitset<N>::which_bit(size_type pos) noexcept
{
    return pos % bits_per_word_;
}

//---------------------------------------------------------------------------//
//! Create a mask for a given bit index
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto Bitset<N>::mask(size_type pos) noexcept
    -> word_type
{
    return word_type(1) << Bitset::which_bit(pos);
}

//---------------------------------------------------------------------------//
/*!
 * Create a negative mask (a single 0 bit) for a given bit index. The purpose
 * of this function is to cast a potentially promoted word_type (from ~) back
 * to the original word_type.
 */
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto Bitset<N>::neg_mask(size_type pos) noexcept
    -> word_type
{
    return ~(word_type(1) << Bitset::which_bit(pos));
}

//---------------------------------------------------------------------------//
//! Get the word for a given bit position
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto Bitset<N>::get_word(size_type pos) const
    noexcept(!CELERITAS_DEBUG) -> word_type
{
    CELER_EXPECT(pos < N);
    return words_[Bitset::which_word(pos)];
}

//---------------------------------------------------------------------------//
//! Get the word for a given bit position
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto
Bitset<N>::get_word(size_type pos) noexcept(!CELERITAS_DEBUG) -> word_type&
{
    CELER_EXPECT(pos < N);
    return words_[Bitset::which_word(pos)];
}

//---------------------------------------------------------------------------//
//! Get the last word of the bitset
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto Bitset<N>::last_word() noexcept -> word_type&
{
    return words_[num_words_ - 1];
}

//---------------------------------------------------------------------------//
//! Get the last word of the bitset
template<size_type N>
CELER_CONSTEXPR_FUNCTION auto Bitset<N>::last_word() const noexcept
    -> word_type
{
    return words_[num_words_ - 1];
}

//---------------------------------------------------------------------------//
//! Clear unused bits in the last word
template<size_type N>
CELER_CONSTEXPR_FUNCTION void Bitset<N>::sanitize() noexcept
{
    constexpr size_type extra_bits = N % bits_per_word_;
    if constexpr (extra_bits != 0)
    {
        this->last_word() &= static_cast<word_type>(
            ~(static_cast<word_type>(~word_type(0)) << extra_bits));
    }
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Bitwise AND
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>
operator&(Bitset<N> const& lhs, Bitset<N> const& rhs)
{
    return Bitset{lhs} &= rhs;
}

//---------------------------------------------------------------------------//
//! Bitwise OR
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>
operator|(Bitset<N> const& lhs, Bitset<N> const& rhs)
{
    return Bitset{lhs} |= rhs;
}

//---------------------------------------------------------------------------//
//! Bitwise XOR
template<size_type N>
CELER_CONSTEXPR_FUNCTION Bitset<N>
operator^(Bitset<N> const& lhs, Bitset<N> const& rhs)
{
    return Bitset{lhs} ^= rhs;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
