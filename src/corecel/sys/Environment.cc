//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 */
std::string const& getenv(std::string const& key)
{
    std::lock_guard<std::mutex> scoped_lock{getenv_mutex()};
    return environment()[key];
}

//---------------------------------------------------------------------------//
/*!
 * Get a true/false flag with a default value.
 *
 * The return value is a combination of the transformed environment value,
 * and a flag specifying whether the environment variable was inserted or
 * otherwise uses the default value.
 */
GetenvFlagResult getenv_flag(std::string const& key, bool default_val)
{
    std::lock_guard<std::mutex> scoped_lock{getenv_mutex()};
    bool inserted = environment().insert({key, default_val ? "1" : "0"});
    if (inserted)
    {
        return {default_val, true};
    }
    std::string const& result = environment()[key];
    if (result.empty())
    {
        // Variable was queried but is empty
        return {default_val, true};
    }

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
