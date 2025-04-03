//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/DiagnosticRngEngine.hh
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
        ++count_;
        return engine_();
    }

    //! Get the number of samples (DEPRECATED: use exchange_count)
    size_type count() const { return count_; }
    //! Reset the sample counter (DEPRECATED: use exchange_count)
    void reset_count() { count_ = 0; }
    //! Get and reset the counter
    size_type exchange_count() { return std::exchange(count_, 0); }

    //!@{
    //! Forwarded functions
    static constexpr result_type min() { return Engine::min(); }
    static constexpr result_type max() { return Engine::max(); }
    //!@}

  private:
    Engine engine_;
    size_type count_ = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
