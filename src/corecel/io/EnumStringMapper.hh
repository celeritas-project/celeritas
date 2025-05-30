//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/EnumStringMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"

#include "detail/EnumStringMapperImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map enums to strings for user output.
 *
 * This should generally be a static const class.
 *
 * Example:
 * \code
    char const* to_cstring(Foo value)
    {
        static EnumStringMapper<Foo> const convert{"foo", "bar", "baz"};
        return convert(value);
    }
 * \endcode
 */
template<class T>
class EnumStringMapper
{
    static_assert(std::is_enum<T>::value, "not an enum type");
    static_assert(static_cast<int>(T::size_) >= 0, "invalid enum type");

  public:
    //! Construct with one string per valid enum value
    template<class... Ts>
    explicit CELER_CONSTEXPR_FUNCTION EnumStringMapper(Ts... strings)
        : strings_{strings..., detail::g_invalid_string}
    {
        // Protect against leaving off a string
        static_assert(sizeof...(strings) == static_cast<size_type>(T::size_),
                      "All strings (except size_) must be explicitly given");
    }

    // Convert from a string
    inline char const* operator()(T value) const;

  private:
    using size_type = std::underlying_type_t<T>;

    Array<char const*, static_cast<size_type>(T::size_) + 1> const strings_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Convert an enum to the corresponding string.
 */
template<class T>
char const* EnumStringMapper<T>::operator()(T value) const
{
    CELER_EXPECT(value <= T::size_);
    return strings_[static_cast<size_type>(value)];
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Write an enum to a stream
template<class T>
auto operator<<(std::ostream& os, T value) -> std::enable_if_t<
    std::is_enum_v<T> && std::is_same_v<decltype(to_cstring(value)), char const*>,
    std::ostream&>
{
    return os << to_cstring(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
