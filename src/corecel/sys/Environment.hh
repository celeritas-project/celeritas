//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Environment.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    //! \name Type aliases
    using key_type = Container::key_type;
    using mapped_type = Container::mapped_type;
    using value_type = Container::value_type;
    using const_iterator = Container::const_iterator;
    using VecKVRef = std::vector<std::reference_wrapper<value_type>>;
    //!@}

  public:
    // Construct with defaults
    Environment() = default;

    // Get an environment variable from current or system environments
    inline mapped_type const& operator[](key_type const&);

    // Insert possibly new environment variables (not thread-safe)
    void insert(value_type const& value);

    //! Get an ordered (by access) vector of key/value pairs
    VecKVRef const& ordered_environment() const { return ordered_; }

    // Remove all entries
    void clear();

    // Insert (not overriding!) from another environment
    void merge(Environment const& other);

  private:
    std::unordered_map<key_type, mapped_type> vars_;
    VecKVRef ordered_;

    mapped_type const& load_from_getenv(key_type const&);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Access a static global environment variable
Environment& environment();

// Thread-safe access to environment variables
std::string const& getenv(std::string const& key);

// Write the accessed environment variables to a stream
std::ostream& operator<<(std::ostream&, Environment const&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get an environment variable from current or system enviroments.
 */
auto Environment::operator[](key_type const& env_var) -> mapped_type const&
{
    auto iter = vars_.find(env_var);
    if (iter == vars_.end())
    {
        return this->load_from_getenv(env_var);
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
