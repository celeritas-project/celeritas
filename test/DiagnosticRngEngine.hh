//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DiagnosticRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <utility>

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostic wrapper that counts the number of calls to operator().
 *
 * This wraps a low-level pseudorandom generator's call function. It can be
 * used to determine the efficiency of rejection algorithms without changing
 * any implementations.
 */
template<class Engine>
class DiagnosticRngEngine
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = typename Engine::result_type;
    using size_type = std::size_t;
    //!@}

  public:
    //! Forward construction arguments to the original engine
    template<class... Args>
    DiagnosticRngEngine(Args&&... args) : engine_(std::forward<Args>(args)...)
    {
    }

    //! Get a random number and increment the sample counter
    result_type operator()()
    {
        ++num_samples_;
        return engine_();
    }

    //! Get the number of samples
    size_type count() const { return num_samples_; }
    //! Reset the sample counter
    void reset_count() { num_samples_ = 0; }

    //!@{
    //! Forwarded functions
    static constexpr result_type min() { return Engine::min(); }
    static constexpr result_type max() { return Engine::max(); }
    //!@}

  private:
    Engine engine_;
    size_type num_samples_ = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
