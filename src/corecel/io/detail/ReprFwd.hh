//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprFwd.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
struct DefaultRepr;

}  // namespace detail

template<class T>
detail::DefaultRepr<T> repr(T const& obj);

}  // namespace celeritas
