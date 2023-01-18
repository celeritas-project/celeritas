//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Environment.cc
//---------------------------------------------------------------------------//
#include "Environment.hh"

#include <cstdlib>
#include <mutex>

#include "corecel/Assert.hh"

namespace celeritas
{
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
    static Environment environ;
    return environ;
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe access to global modified environment variables.
 */
std::string const& getenv(std::string const& key)
{
    static std::mutex getenv_mutex;
    std::lock_guard<std::mutex> scoped_lock{getenv_mutex};
    return environment()[key];
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
    auto iter_inserted = vars_.emplace(key, std::move(value));
    CELER_ASSERT(iter_inserted.second);
    ordered_.push_back(std::ref(*iter_inserted.first));

    CELER_ENSURE(ordered_.size() == vars_.size());
    return iter_inserted.first->second;
}
//---------------------------------------------------------------------------//
/*!
 * Set environment variables en masse.
 *
 * Existing environment variables will *not* be overwritten.
 */
void Environment::insert(value_type const& value)
{
    auto iter_inserted = vars_.insert(value);
    if (iter_inserted.second)
    {
        ordered_.push_back(std::ref(*iter_inserted.first));
    }
    CELER_ENSURE(ordered_.size() == vars_.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
