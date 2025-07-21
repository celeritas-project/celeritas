//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2025 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/compare_benchmarks.C
//! \brief Compare DD4hep simulation data between two configurations
//---------------------------------------------------------------------------//
/*!
 * Usage:
 * $ root
 * root[0] .x compare_benchmarks.C("config1-out.root", "config2-out.root")
 *
 * DD4hep simulation output comparison for validation studies.
 *
 * Update histogram info and select data accordingly in the Helper functions
 * and static variables section.
 *
 * Plot attributes are meant to be used with the Celeritas plot style. See
 * https://github.com/celeritas-project/benchmarks/blob/main/rootlogon.C
 */
//---------------------------------------------------------------------------//
#include <vector>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TMath.h>
#include <TText.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

// Include dd4hep headers
#include "DDG4/Geant4Data.h"
#include "DDG4/Geant4Particle.h"

// Include additional ROOT headers for 2D plotting
#include <TH2D.h>
#include <TLatex.h>
#include <TPad.h>
#include <TStyle.h>

//---------------------------------------------------------------------------//
//! Helper functions and static variables
//---------------------------------------------------------------------------//
// Histogram definition
static int const n_bins = 50;
static double const bin_min = 0;
static double const bin_max = 50;
static TString const hist_title
    = "MC Particle p_{T} Distribution (DD4hep Simulation)";
static TString const commit_hash = "";
static TString const x_axis_title = "p_{T} [GeV]";
static TString const config1_legend = "Configuration 1";
static TString const config2_legend = "Configuration 2";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Loop over events for a given ROOT file and populate histogram
void loop(TString file, TH1D* hist, TString plot_type)
{
    std::cout << "Processing " << file.Data() << std::endl;
    auto tfile = TFile::Open(file.Data(), "read");
    if (!tfile || tfile->IsZombie())
    {
        std::cerr << "Error: Cannot open file " << file.Data() << std::endl;
        return;
    }

    // Create TTreeReader
    TTreeReader reader("EVENT", tfile);

    // Define TTreeReaderValues for different branches
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Particle*>> mcParticles(
        reader, "MCParticles");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>> trackerHits(
        reader, "TrackerHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>
        calorimeterHits(reader, "CalorimeterHits");

    // Event loop
    while (reader.Next())
    {
        if (plot_type == "pt" || plot_type == "eta" || plot_type == "phi")
        {
            // Analyze MC particles
            for (auto const& particle : *mcParticles)
            {
                if (particle)
                {
                    // Get momentum components
                    double px = particle->psx;
                    double py = particle->psy;
                    double pz = particle->psz;

                    if (plot_type == "pt")
                    {
                        double pt = TMath::Sqrt(px * px + py * py);
                        hist->Fill(pt);
                    }
                    else if (plot_type == "eta")
                    {
                        double p = TMath::Sqrt(px * px + py * py + pz * pz);
                        if (p > 0 && TMath::Abs(pz) < p)
                        {
                            double theta = TMath::ACos(pz / p);
                            if (theta > 0 && theta < TMath::Pi())
                            {
                                double eta
                                    = -TMath::Log(TMath::Tan(0.5 * theta));
                                hist->Fill(eta);
                            }
                        }
                    }
                    else if (plot_type == "phi")
                    {
                        double phi = TMath::ATan2(py, px);
                        hist->Fill(phi);
                    }
                }
            }
        }
        else if (plot_type == "calo_energy")
        {
            // Analyze calorimeter hits
            for (auto const& hit : *calorimeterHits)
            {
                if (hit)
                {
                    double energy = hit->energyDeposit;
                    hist->Fill(energy);
                }
            }
        }
    }

    tfile->Close();
}

//---------------------------------------------------------------------------//
/*!
 * Main function.
 */
void compare_1D_histos(TString config1_rootfile,
                       TString config2_rootfile,
                       TString plot_type)
{
    // Update histogram parameters based on plot variable
    double hist_bin_min = bin_min;
    double hist_bin_max = bin_max;
    TString hist_x_axis = x_axis_title;
    TString hist_plot_title = hist_title;

    if (plot_type == "eta")
    {
        hist_bin_min = 0;
        hist_bin_max = 5;
        hist_x_axis = "#eta";
        hist_plot_title = "MC Particle #eta Distribution (DD4hep Simulation)";
    }
    else if (plot_type == "phi")
    {
        hist_bin_min = -TMath::Pi();
        hist_bin_max = TMath::Pi();
        hist_x_axis = "#phi";
        hist_plot_title = "MC Particle #phi Distribution (DD4hep Simulation)";
    }
    else if (plot_type == "calo_energy")
    {
        hist_bin_min = 0;
        hist_bin_max = 10;
        hist_x_axis = "Energy [GeV]";
        hist_plot_title
            = "Calorimeter Hit Energy Distribution (DD4hep Simulation)";
    }

    auto h_config1
        = new TH1D("Config1", "", n_bins, hist_bin_min, hist_bin_max);
    auto h_config2
        = new TH1D("Config2", "", n_bins, hist_bin_min, hist_bin_max);

    // Process data
    loop(config1_rootfile, h_config1, plot_type);
    loop(config2_rootfile, h_config2, plot_type);

    // Create relative error histograms
    auto h_config1_rel_err = new TH1D(
        "Config1 rel. err.", "", n_bins, hist_bin_min, hist_bin_max);
    auto h_config1_rel_err_3s = new TH1D(
        "Config1 rel. err. 3sigma", "", n_bins, hist_bin_min, hist_bin_max);

    for (int i = 0; i < n_bins; i++)
    {
        double error = h_config1->GetBinError(i);
        double value = h_config1->GetBinContent(i);
        double rel_err = value ? error / value : 0;

        h_config1_rel_err->SetBinContent(i, 0);
        h_config1_rel_err->SetBinError(i, rel_err * 100);
        h_config1_rel_err_3s->SetBinContent(i, 0);
        h_config1_rel_err_3s->SetBinError(i, 3 * rel_err * 100);
    }

    // Create relative difference histogram [(Config1 - Config2) / Config1]
    auto h_rel_diff = (TH1D*)h_config1->Clone();
    h_rel_diff->Add(h_config2, -1);
    h_rel_diff->Divide(h_config1);
    h_rel_diff->Scale(100);  // In [%]

    // Create canvas
    auto canvas = new TCanvas("c1", "c1", 750, 600);
    canvas->Divide(1, 2);

    // Create top pad and move to it
    auto pad_top = new TPad("pad1", "", 0.0, 0.3, 1.0, 1.0);
    pad_top->SetBottomMargin(0.02);
    pad_top->SetLeftMargin(0.11);
    pad_top->Draw();
    pad_top->cd();

    // Histograms attributes
    auto const config2_color = kAzure + 1;
    h_config2->SetLineColor(config2_color);
    h_config2->SetLineWidth(2);
    h_config1->SetMarkerStyle(46);
    h_config1->SetMarkerSize(1.6);

    h_config1->GetXaxis()->SetLabelOffset(99);
    h_config1->GetYaxis()->SetLabelOffset(0.007);
    h_config1->GetYaxis()->CenterTitle();

    // Draw histograms
    h_config1->Draw("PE2");
    h_config2->Draw("hist sames");

    auto legend_top = new TLegend(0.57, 0.46, 0.86, 0.86);
    legend_top->AddEntry(h_config1, config1_legend, "p");
    legend_top->AddEntry(h_config2, config2_legend, "l");
    legend_top->AddEntry(new TH1D(), "Statistical errors:", "f");
    legend_top->AddEntry(h_config1_rel_err, "1#sigma", "f");
    legend_top->AddEntry(h_config1_rel_err_3s, "3#sigma", "f");
    legend_top->SetMargin(0.27);
    legend_top->SetLineColor(kGray);
    legend_top->Draw();

    auto title_text = new TText(0.17, 0.92, hist_plot_title);
    title_text->SetNDC();
    title_text->SetTextColor(kGray);
    title_text->Draw();

    auto commit_text = new TLatex(0.67, 0.92, commit_hash);
    commit_text->SetNDC();
    commit_text->SetTextColor(kGray);
    commit_text->Draw();

    // Redraw axis above the histogram lines
    pad_top->RedrawAxis();
    pad_top->SetLogy();
    // Move back to canvas
    canvas->cd();

    // Create bottom pad and move to it
    auto pad_bottom = new TPad("pad2", "", 0.0, 0.0, 1.0, 0.3);
    pad_bottom->SetTopMargin(0.02);
    pad_bottom->SetBottomMargin(0.33);
    pad_bottom->SetLeftMargin(0.11);
    pad_bottom->Draw();
    pad_bottom->cd();

    h_config1_rel_err_3s->GetXaxis()->SetTitle(hist_x_axis);
    h_config1_rel_err_3s->GetXaxis()->CenterTitle();
    h_config1_rel_err_3s->GetXaxis()->SetTitleSize(0.14);
    h_config1_rel_err_3s->GetXaxis()->SetTitleOffset(1.1);
    h_config1_rel_err_3s->GetXaxis()->SetLabelSize(0.1153);
    h_config1_rel_err_3s->GetXaxis()->SetLabelOffset(0.02);
    h_config1_rel_err_3s->GetXaxis()->SetTickLength(0.07);

    h_config1_rel_err_3s->GetYaxis()->SetTitle("Rel. Diff. (%)");
    h_config1_rel_err_3s->GetYaxis()->CenterTitle();
    h_config1_rel_err_3s->GetYaxis()->SetTitleSize(0.131);
    h_config1_rel_err_3s->GetYaxis()->SetTitleOffset(0.415);
    h_config1_rel_err_3s->GetYaxis()->SetLabelSize(0.116);
    h_config1_rel_err_3s->GetYaxis()->SetLabelOffset(0.008);
    h_config1_rel_err_3s->GetYaxis()->SetTickLength(0.04);
    h_config1_rel_err_3s->GetYaxis()->SetNdivisions(503);
    h_config1_rel_err_3s->GetYaxis()->SetRangeUser(-5, 5);

    h_config1_rel_err_3s->SetLineColorAlpha(kGray, 0.7);
    h_config1_rel_err_3s->SetFillColorAlpha(kGray, 0.7);
    h_config1_rel_err_3s->SetMarkerSize(0);
    h_config1_rel_err->SetLineColorAlpha(kGray + 1, 0.7);
    h_config1_rel_err->SetFillColorAlpha(kGray + 1, 0.7);
    h_config1_rel_err->SetMarkerSize(0);
    h_config1_rel_err->GetYaxis()->SetRangeUser(-5, 5);
    h_rel_diff->SetLineColor(config2_color);

    // Draw stat. err. and rel. diff. histograms
    h_config1_rel_err_3s->Draw("hist E2");
    h_config1_rel_err->Draw("hist E2 sames");
    h_rel_diff->Draw("hist sames");

    pad_bottom->RedrawAxis();
    canvas->SetLogy();
    canvas->Print(config1_legend + "_" + config2_legend + "_" + plot_type
                  + ".png");

    // Clean up
    delete canvas;
    delete h_config1;
    delete h_config2;
    delete h_config1_rel_err;
    delete h_config1_rel_err_3s;
    delete h_rel_diff;
}

//---------------------------------------------------------------------------//
// Helper functions for 2D histogram comparison
//---------------------------------------------------------------------------//
void SetAxisStyle(TAxis* axis,
                  double titleSize,
                  double titleOffset,
                  double labelSize,
                  double labelOffset,
                  double tickLength,
                  int nDivisions,
                  bool isXaxis)
{
    if (isXaxis)
    {
        axis->SetMaxDigits(4);
    }
    axis->SetTitleSize(titleSize);
    axis->SetTitleOffset(titleOffset);
    axis->SetLabelSize(labelSize);
    axis->SetLabelOffset(labelOffset);
    axis->SetTickLength(tickLength);
    axis->SetNdivisions(nDivisions);
    return;
}

void DrawEntryNote(TCanvas* c, TH2D* h, int pad_num, TString primary)
{
    TLatex* text = new TLatex(0.37, 0.88, Form("%s", primary.Data()));
    text->SetTextSize(0.04);
    text->SetNDC();
    c->cd(pad_num);
    text->Draw();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Create 2D histograms from DD4hep data
void create_2D_histograms(TString file, TH2D* h, TString plot_type)
{
    std::cout << "Processing 2D data from " << file.Data() << std::endl;
    auto tfile = TFile::Open(file.Data(), "read");
    if (!tfile || tfile->IsZombie())
    {
        std::cerr << "Error: Cannot open file " << file.Data() << std::endl;
        return;
    }

    // Create TTreeReader
    TTreeReader reader("EVENT", tfile);

    // Define TTreeReaderValues for different branches
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>> trackerHits(
        reader, "TrackerHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>
        calorimeterHits(reader, "CalorimeterHits");

    // Event loop
    while (reader.Next())
    {
        // Fill tracker r-z histogram
        if (plot_type == "tracker_rz")
        {
            for (auto const& hit : *trackerHits)
            {
                if (hit)
                {
                    double x = hit->position.x();
                    double y = hit->position.y();
                    double z = hit->position.z();
                    double r = TMath::Sqrt(x * x + y * y);
                    h->Fill(z, r);
                }
            }
        }

        if (plot_type == "calo_xy")
        {
            // Fill calorimeter x-y histogram
            for (auto const& hit : *calorimeterHits)
            {
                if (hit)
                {
                    double x = hit->position.x();
                    double y = hit->position.y();
                    h->Fill(x, y);
                }
            }
        }
    }

    tfile->Close();
}

//---------------------------------------------------------------------------//
/*!
 * 2D histogram comparison function
 */
int compare_2D_histos(TString f1_name,
                      TString f2_name,
                      TString h1_label,
                      TString h2_label,
                      TString plot_type,
                      TString primary)
{
    // Create histograms based on plot type
    TH2D *h1, *h2;

    if (plot_type == "calo_xy")
    {
        h1 = new TH2D("h1_calo_xy",
                      "Calorimeter Hit Position;X [mm];Y [mm]",
                      200,
                      -2000,
                      2000,
                      200,
                      -2000,
                      2000);
        h2 = new TH2D("h2_calo_xy",
                      "Calorimeter Hit Position;X [mm];Y [mm]",
                      200,
                      -2000,
                      2000,
                      200,
                      -2000,
                      2000);

        // Fill histograms from DD4hep data
        create_2D_histograms(f1_name, h1, plot_type);
        create_2D_histograms(f2_name, h2, plot_type);
    }
    else if (plot_type == "tracker_rz")
    {
        h1 = new TH2D("h1_tracker_rz",
                      "Tracker Hit r-z Distribution;z [mm];r [mm]",
                      200,
                      -3000,
                      3000,
                      100,
                      0,
                      2000);
        h2 = new TH2D("h2_tracker_rz",
                      "Tracker Hit r-z Distribution;z [mm];r [mm]",
                      200,
                      -3000,
                      3000,
                      100,
                      0,
                      2000);

        // Fill histograms from DD4hep data
        create_2D_histograms(f1_name, h1, plot_type);
        create_2D_histograms(f2_name, h2, plot_type);
    }
    else
    {
        std::cerr << "Error: Unknown plot type " << plot_type.Data()
                  << std::endl;
        return 1;
    }

    // Initialize canvas
    TCanvas* c = new TCanvas("c", "c", 1200, 600);
    c->Divide(2, 1);

    // Turn on Stat Box and position it
    gStyle->SetOptStat(1);
    gStyle->SetStatX(0.85);
    gStyle->SetStatY(0.85);

    // Switch to pad 1 and draw histogram from first file
    c->cd(1);
    h1->SetTitle(h1_label);
    h1->Draw("colz");

    // Set z-axis range and log scale
    h1->SetMaximum(1e5);
    h1->SetMinimum(1);
    gPad->SetLogz();

    // Set margins on pad and style the axes
    gPad->SetMargin(0.13, 0.13, 0.13, 0.13);
    SetAxisStyle(
        h1->GetXaxis(), 0.06, 0.415, 0.04, 0.02, 0.04, 3015, true);  // X axis
    SetAxisStyle(
        h1->GetYaxis(), 0.06, 0.415, 0.04, 0.02, 0.04, 1006, false);  // Y axis

    // Draw additional information label below histogram title
    DrawEntryNote(c, h1, 1, primary);

    // Switch to pad 2 and draw histogram from second file
    c->cd(2);
    h2->SetTitle(h2_label);
    h2->Draw("colz");

    // Set z-axis range and log scale
    h2->SetMaximum(1e5);
    h2->SetMinimum(1);
    gPad->SetLogz();

    // Set margins on pad and style the axes
    gPad->SetMargin(0.13, 0.13, 0.13, 0.13);
    SetAxisStyle(
        h2->GetXaxis(), 0.04, 1.6, 0.04, 0.02, 0.04, 3015, true);  // X axis
    SetAxisStyle(
        h2->GetYaxis(), 0.04, 1.6, 0.04, 0.02, 0.04, 1006, false);  // Y axis

    // Draw additional information label below histogram title
    DrawEntryNote(c, h2, 2, primary);

    // Save image with appropriate name
    c->Print(h1_label + "_" + h2_label + "_" + plot_type + ".png");

    // Clean up
    delete c;
    delete h1;
    delete h2;

    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Wrapper function for backward compatibility and demonstration
 */
void compare_benchmarks(TString config1_rootfile, TString config2_rootfile)
{
    std::cout << "Running 1D histogram comparison..." << std::endl;
    compare_1D_histos(config1_rootfile, config2_rootfile, "pt");
    compare_1D_histos(config1_rootfile, config2_rootfile, "eta");
    compare_1D_histos(config1_rootfile, config2_rootfile, "phi");
    compare_1D_histos(config1_rootfile, config2_rootfile, "calo_energy");

    std::cout << "Running 2D histogram comparisons..." << std::endl;
    // Example 2D comparisons
    compare_2D_histos(config1_rootfile,
                      config2_rootfile,
                      "Configuration 1",
                      "Configuration 2",
                      "calo_xy",
                      "DD4hep Calorimeter Analysis");

    compare_2D_histos(config1_rootfile,
                      config2_rootfile,
                      "Configuration 1",
                      "Configuration 2",
                      "tracker_rz",
                      "DD4hep Tracker Analysis");
}
