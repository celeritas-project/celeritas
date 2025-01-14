//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringEnumMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map strings to enums for user input.
 *
 * Note that since a map is built at construction time, instances of this class
 * should be \c static to amortize the cost. The strings being converted *must*
 * exceed the lifetime of this class. (Usually it references `const char*`.)
 *
 * \todo If the size of the strings is short and there aren't a lot of them, it
 * will be faster to use a fixed-size array and search over them.
 *
 * Example:
 * \code
void from_json(const nlohmann::json& j, GeantSetupOptions& opts)
{
    static auto gspl_from_string
        = StringEnumMapper<PhysicsList>::from_cstring_func(
            to_cstring, "physics list");
    opts.physics = gspl_from_string(j.at("physics").get<std::string>());
}
   \endcode
 */
template<class T>
class StringEnumMapper
{
    static_assert(std::is_enum<T>::value, "not an enum type");
    static_assert(static_cast<int>(T::size_) >= 0, "invalid enum type");

  public:
    //!@{
    //! \name Type aliases
    using EnumCStringFuncPtr = char const*(T);
    //!@}

  public:
    // Construct from a "to_cstring" function pointer
    static inline StringEnumMapper<T>
    from_cstring_func(EnumCStringFuncPtr, char const* desc = nullptr);

    // Construct with a function that takes an enum and returns a stringlike
    template<class U>
    // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
    explicit inline StringEnumMapper(U&& enum_to_string,
                                     char const* desc = nullptr);

    // Convert from a string
    inline T operator()(std::string_view s) const;

  private:
    char const* description_;
    std::unordered_map<std::string_view, T> map_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct using a \c to_cstring function.
 */
template<class T>
StringEnumMapper<T>
StringEnumMapper<T>::from_cstring_func(EnumCStringFuncPtr fp, char const* desc)
{
    CELER_EXPECT(fp);
    return StringEnumMapper<T>{fp, desc};
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a "stringify" function.
 *
 * The result just has to be implicitly convertible to a \c std::string_view .
 */
template<class T>
template<class U>
// NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
StringEnumMapper<T>::StringEnumMapper(U&& enum_to_string, char const* desc)
    : description_(desc)
{
    map_.reserve(static_cast<std::size_t>(T::size_));
    for (auto v : celeritas::range(T::size_))
    {
        auto iter_inserted = map_.insert({enum_to_string(v), v});
        CELER_ASSERT(iter_inserted.second);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a string_view to the corresponding enum.
 */
template<class T>
T StringEnumMapper<T>::operator()(std::string_view s) const
{
    auto result = map_.find(s);
    CELER_VALIDATE(result != map_.end(),
                   << "invalid " << (description_ ? description_ : "value")
                   << " '" << s << '\'');
    return result->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
