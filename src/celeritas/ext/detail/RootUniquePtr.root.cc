//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/RootUniquePtr.root.cc
//---------------------------------------------------------------------------//
#include "RootUniquePtr.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void RootDeleter<T>::operator()(T* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template struct RootDeleter<TFile>;
template struct RootDeleter<TTree>;
template struct RootDeleter<TBranch>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
