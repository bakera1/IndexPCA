@unittest.skip
@timeit
def test_simple_index_extract_with_export_to_csv(self):
    """

        simple test that writes out a csv file

        CDX NAHYB,CDX-NAHYBS29V1-5Y,1.00481876
        CDX NAHYBB,CDX-NAHYBBS29V1-5Y,1.16801886
        CDXEM,CDX-EMS28V1-10Y,0.89080159
        CDXEM,CDX-EMS28V1-5Y,0.9646434
        CDXEMDIV,CDX-EMDIVS12V3-10Y,0.983255

    :return:
    """

    # Step 1. extract risk data from db

    self.reference_date = '20171016'
    self.myfilename = os.path.join(os.getcwd(), 'example.{0}.csv'.format(self.reference_date))
    with session_scope() as session:
        query = session.query(MarkITCreditIndex)\
                .filter(MarkITCreditIndex.business_date == self.reference_date)\
                .filter(MarkITCreditIndex.on_the_run == 'Y')\
                .order_by(MarkITCreditIndex.fido_instance_id.asc())

        # build a data frame
        df = data_frame(query, [c for c in attribute_names(MarkITCreditIndex)])

        # some example aggregate on a group by in wonderful pandas
        df.groupby(['name', 'index_id'])['model_price']\
            .aggregate(lambda grp: grp.mean()).to_csv(self.myfilename)

        # check we have exported.
        self.assertTrue(os.path.isfile(self.myfilename))