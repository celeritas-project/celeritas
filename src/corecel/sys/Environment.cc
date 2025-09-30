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
#include "corecel/io/StringUtils.hh"

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
 * As with the general \c Environment instance that this references, any
 * already-set values (e.g., from JSON input) override whatever variables are
 * in the system environment (e.g., from the shell script that invoked this
 * executable).
 *
 * - Allowed true values: <code>"1", "t", "yes", "true", "True"</code>
 * - Allowed false values: <code>"0", "f", "no", "false", "False"</code>
 * - Empty value returns the default
 * - Other value warns and returns the default
 */
GetenvFlagResult getenv_flag(std::string const& key, bool default_val)
{
    std::scoped_lock lock_{getenv_mutex()};

    // Get the string value from the existing environment *or* system
    std::string str_value = [&key]() -> std::string {
        auto& env = environment();
        if (auto iter = env.find(key); iter != env.end())
        {
            // Variable was already loaded internally
            if (iter->second.empty())
            {
                CELER_LOG(warning)
                    << "Already-set but empty environment value '" << key
                    << "' is being ignored";
            }
            return iter->second;
        }
        else if (char const* sys_value = std::getenv(key.c_str()))
        {
            // Variable is set in the user environment
            return sys_value;
        }
        return {};
    }();

    GetenvFlagResult result;
    result.defaulted = str_value.empty();
    result.value = default_val;
    if (!result.defaulted)
    {
        str_value = tolower(str_value);

        static char const* const true_str[] = {"1", "t", "yes", "true"};
        static char const* const false_str[] = {"0", "f", "no", "false"};

        if (std::find(std::begin(true_str), std::end(true_str), str_value)
            != std::end(true_str))
        {
            result.value = true;
        }
        else if (std::find(std::begin(false_str), std::end(false_str), str_value)
                 != std::end(false_str))
        {
            result.value = false;
        }
        else
        {
            CELER_LOG(warning) << "Invalid environment value " << key << "="
                               << result.value << ": expected a flag";
        }
    }
    else
    {
        // Save string value to be added to environment
        str_value = default_val ? "1" : "0";
    }

    environment().insert({key, str_value});
    return result;
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
 *
 * \deprecated Use `env = {}` instead.
 */
void Environment::clear()
{
    *this = {};
}

//---------------------------------------------------------------------------//
/*!
 * Insert but don't override from another environment.
 */
void Environment::merge(Environment const& other)
{
    for (auto const& kv : other.ordered_environment())
    {
        auto inserted = this->insert(kv);
        if (!inserted)
        {
            auto&& [key, val] = kv.get();
            auto const& existing = vars_.at(key);
            if (val != existing)
            {
                CELER_LOG(warning)
                    << "Ignoring new environment variable " << key << "="
                    << val << ": using existing value '" << existing << "'";
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
