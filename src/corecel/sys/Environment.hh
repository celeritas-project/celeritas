//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * This makes it easier to generate reproducible runs, launch Celeritas
 * remotely, or integrate with application drivers. The environment variables
 * may be encoded as JSON input to supplement or override system environment
 * variables, or set programmatically via this API call. Later the environment
 * class can be interrogated to find which environment variables were accessed.
 *
 * Unlike the standard environment which returns a null pointer for an *unset*
 * variable, this returns an empty string.
 *
 * \note This class is not thread-safe on its own. The \c celeritas::getenv
 * free function however is safe, although it should only be used in setup
 * (single-thread) steps.
 *
 * \note Once inserted into the environment map, values cannot be changed.
 * Standard practice in the code is to evaluate the environment variable
 * exactly \em once and cache the result as a static const variable. If you
 * really wanted to, you could call <code> celeritas::environment() = {};
 * </code> but that could result in the end-of-run diagnostic reporting
 * different values than the ones actually used during the code's setup.
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

    // Determine whether a variable has already been set (mostly internal)
    inline const_iterator find(key_type const&) const;

    // Insert possibly new environment variables
    bool insert(value_type const& value);

    //! Get an ordered (by access) vector of key/value pairs
    VecKVRef const& ordered_environment() const { return ordered_; }

    // Remove all entries
    void clear();

    // Insert (not overriding!) from another environment
    void merge(Environment const& other);

    //!@{
    //! Access all entries, unordered, by const iterator
    const_iterator begin() const { return vars_.cbegin(); }
    const_iterator cbegin() const { return vars_.cbegin(); }
    const_iterator end() const { return vars_.cend(); }
    const_iterator cend() const { return vars_.cend(); }
    //!@}

  private:
    std::unordered_map<key_type, mapped_type> vars_;
    VecKVRef ordered_;

    mapped_type const& load_from_getenv(key_type const&);
};

//---------------------------------------------------------------------------//
//! Return result from \c getenv_flag
struct GetenvFlagResult
{
    bool value{};  //!< Determined by user or default
    bool defaulted{};  //!< True if no valid user value was present
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Access a static global environment variable
Environment& environment();

// Thread-safe access to environment variables
std::string const& getenv(std::string const& key);

// Thread-safe flag access to environment variables
GetenvFlagResult getenv_flag(std::string const& key, bool default_val);

// Thread-safe flag access to environment variables with lazy function default
GetenvFlagResult
getenv_flag_lazy(std::string const& key, std::function<bool()> const&);

// Write the accessed environment variables to a stream
std::ostream& operator<<(std::ostream&, Environment const&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Determine whether a variable has already been set and get its value if so.
 */
auto Environment::find(key_type const& env_var) const -> const_iterator
{
    return vars_.find(env_var);
}

//---------------------------------------------------------------------------//
/*!
 * Get an environment variable from current or system environments.
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
