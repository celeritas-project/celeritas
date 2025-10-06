//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TypeDemangler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <typeinfo>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Utility function for demangling C++ types (specifically with GCC).
 *
 * See:
 * http://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
 * Example:
 * \code
    std::string int_type = demangled_typeid_name(typeid(int).name());
    TypeDemangler<Base> demangle;
    std::string static_type = demangle();
    std::string dynamic_type = demangle(Derived());
   \endcode
 */
template<class T>
class TypeDemangler
{
  public:
    // Get the pretty typename of the instantiated type (static)
    inline std::string operator()() const;
    // Get the *dynamic* pretty typename of a variable (dynamic)
    inline std::string operator()(T const&) const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Demangle the name that comes from `typeid`
std::string demangled_typeid_name(char const* typeid_name);

//---------------------------------------------------------------------------//
//! Demangle the type name of any variable
template<class T>
std::string demangled_type(T&& t)
{
    return demangled_typeid_name(typeid(t).name());
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the pretty typename of the instantiated type (static).
 */
template<class T>
std::string TypeDemangler<T>::operator()() const
{
    return demangled_typeid_name(typeid(T).name());
}

//---------------------------------------------------------------------------//
/*!
 * Get the *dynamic* pretty typename of a variable (dynamic).
 */
template<class T>
std::string TypeDemangler<T>::operator()(T const& t) const
{
    return demangled_typeid_name(typeid(t).name());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
