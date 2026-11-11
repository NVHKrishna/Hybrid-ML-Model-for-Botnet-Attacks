 if args.output
        else run_dir / "predictions" / (
            "prediction_record.csv" if args.record is not None
            else f"prediction_id_{args.id}.csv" if args.id is not None
            else f"prediction_row_{args.row}.csv"
        )
    )
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved record and combined prediction to: {output_path}")


if __name__ == "__main__":
    main()
