//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Environment.cc
//---------------------------------------------------------------------------//
#include "Environment.hh"

#include <algorithm>
#include <cstdlib>
#include <mutex>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::mutex& getenv_mutex()
{
    static std::mutex mu;
    return mu;
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Access a static global environment variable.
 *
 * This static variable should be shared among Celeritas objects.
 */
Environment& environment()
{
    static Environment result;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe access to global modified environment variables.
 *
 * This function will \em insert the current value of the key into the
 * environment, which remains immutable over the lifetime of the program
 * (allowing the use of <code>static const</code> data to be set from the
 * environment).
 */
std::string const& getenv(std::string const& key)
{
    std::scoped_lock lock_{getenv_mutex()};
    return environment()[key];
}

//---------------------------------------------------------------------------//
/*!
 * Get a true/false flag with a default value.
 *
 * The return value is a pair that has (1) the flag as determined by the
 * environment variable or default value, and (2) an "insertion" flag
 * specifying whether the default was used. The insertion result can be useful
 * for providing a diagnostic message to the user.
 *
 * - Allowed true values: <code>"1", "t", "yes", "true", "True"</code>
 * - Allowed false values: <code>"0", "f", "no", "false", "False"</code>
 * - Empty value returns the default
 * - Other value warns and returns the default
 */
GetenvFlagResult getenv_flag(std::string const& key, bool default_val)
{
    std::scoped_lock lock_{getenv_mutex()};

    std::string value;
    if (char const* sys_value = std::getenv(key.c_str()))
    {
        // Variable is set in the user environment
        value = sys_value;
    }
    if (value.empty() && environment().insert({key, default_val ? "1" : "0"}))
    {
        // Variable is empty
        return {default_val, true};
    }

    std::string const& result = environment()[key];
    static char const* const true_str[] = {"1", "t", "yes", "true", "True"};
    if (std::find(std::begin(true_str), std::end(true_str), result)
        != std::end(true_str))
    {
        return {true, false};
    }
    static char const* const false_str[] = {"0", "f", "no", "false", "False"};
    if (std::find(std::begin(false_str), std::end(false_str), result)
        != std::end(false_str))
    {
        return {false, false};
    }
    CELER_LOG(warning) << "Invalid environment value " << key << "=" << result
                       << ": expected a flag";
    return {default_val, true};
}

//---------------------------------------------------------------------------//
/*!
 * Write the accessed environment variables to a stream.
 */
std::ostream& operator<<(std::ostream& os, Environment const& env)
{
    os << "{\n";
    for (auto const& kvref : env.ordered_environment())
    {
        Environment::value_type const& kv = kvref;
        os << "  " << kv.first << ": '" << kv.second << "',\n";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Set a value from the system environment.
 */
auto Environment::load_from_getenv(key_type const& key) -> mapped_type const&
{
    std::string value;
    if (char const* sys_value = std::getenv(key.c_str()))
    {
        // Variable is set in the user environment
        value = sys_value;
    }

    // Insert value and ordering. Note that since the elements are never
    // erased, pointers to the keys are guaranteed to always be valid.
    auto [iter, inserted] = vars_.emplace(key, std::move(value));
    CELER_ASSERT(inserted);
    ordered_.push_back(std::ref(*iter));

    CELER_ENSURE(ordered_.size() == vars_.size());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Set a single environment variable that hasn't yet been set.
 *
 * Existing environment variables will *not* be overwritten.
 *
 * \return Whether insertion took place
 */
bool Environment::insert(value_type const& value)
{
    auto [iter, inserted] = vars_.insert(value);
    if (inserted)
    {
        ordered_.push_back(std::ref(*iter));
    }
    CELER_ENSURE(ordered_.size() == vars_.size());
    return inserted;
}

//---------------------------------------------------------------------------//
/*!
 * Remove all entries.
 */
void Environment::clear()
{
    vars_.clear();
    ordered_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Insert but don't override from another environment.
 */
void Environment::merge(Environment const& other)
{
    for (auto const& kv : other.ordered_environment())
    {
        this->insert(kv);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
