//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Environment.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interrogate and extend environment variables.
 *
 * This makes it easier to generate reproducible runs or launch Celeritas
 * remotely: the environment variables may be encoded as JSON input to
 * supplement or override system environment variables. Later it can be
 * interrogated to find which environment variables were accessed.
 *
 * Unlike the standard environment which returns a null pointer for an *unset*
 * variable, this returns an empty string. When we switch to C++17 we can
 * return a `std::optional<std::string>` if this behavior isn't appropriate.
 *
 * \note This class is not thread-safe on its own. The \c celeritas::getenv
 * free function however is safe, although it should only be used in setup
 * (single-thread) steps.
 */
class Environment
{
  private:
    using Container = std::unordered_map<std::string, std::string>;

  public:
    //!@{
    //! Type aliases
    using key_type       = Container::key_type;
    using mapped_type    = Container::mapped_type;
    using value_type     = Container::value_type;
    using const_iterator = Container::const_iterator;
    //!@}

  public:
    // Construct with defaults
    Environment() = default;

    // Get an environment variable from current or system enviroments
    inline const mapped_type& operator[](const key_type&);

    // Insert possibly new environment variables (not thread-safe)
    inline void insert(const value_type& value);

    //!@{
    //! Iterator access to existing variables
    const_iterator cbegin() const { return vars_.cbegin(); }
    const_iterator cend() const { return vars_.cend(); }
    const_iterator begin() const { return vars_.begin(); }
    const_iterator end() const { return vars_.end(); }
    //!@}

  private:
    std::unordered_map<key_type, mapped_type> vars_;

    const mapped_type& load_from_getenv(const key_type&);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Access a static global environment variable
Environment& environment();

// Thread-safe access to environment variables
const std::string& getenv(const std::string& key);

// Write the accessed environment variables to a stream
std::ostream& operator<<(std::ostream&, const Environment&);

//---------------------------------------------------------------------------//
// INLINE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get an environment variable from current or system enviroments.
 */
auto Environment::operator[](const key_type& env_var) -> const mapped_type&
{
    auto iter = vars_.find(env_var);
    if (iter == vars_.end())
    {
        return this->load_from_getenv(env_var);
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Set environment variables en masse.
 *
 * Existing environment variables will *not* be overwritten.
 */
void Environment::insert(const value_type& value)
{
    vars_.insert(value);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
